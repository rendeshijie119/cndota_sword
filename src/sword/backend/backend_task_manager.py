import os
import sys
import json
import uuid
import subprocess
from pathlib import Path
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

# Unified paths (src/sword/data/*)
try:
    from src.sword.shared.paths import DATA_DIR, summary
except Exception:
    repo_root_src = Path(__file__).resolve().parents[2]
    if str(repo_root_src) not in sys.path:
        sys.path.append(str(repo_root_src))
    from src.sword.shared.paths import DATA_DIR, summary

app = FastAPI()

allowed_origins = os.environ.get("ALLOWED_ORIGINS", "*")
origins = [o.strip() for o in allowed_origins.split(",")] if allowed_origins else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

task_status = {}
PROGRESS_PATH = DATA_DIR / "pipeline_progress.json"
RESULT_PATH = DATA_DIR / "pipeline_results.json"

def _repo_root() -> Path:
    """
    parents: backend_task_manager.py -> backend -> sword -> src -> <repo>
    """
    p = Path(__file__).resolve()
    try:
        return p.parents[3]
    except Exception:
        for parent in p.parents:
            if parent.name == "src":
                return parent.parent
        return p.parents[0]

def _resolve_pipeline_script(repo_root: Path) -> Path:
    """
    尝试两种路径：
    1) <repo>/src/sword/backend/all_in_one_pipeline.py
    2) <repo>/backend/pipeline/all_in_one_pipelines.py
    返回第一个存在的路径，否则空路径。
    """
    candidates = [
        repo_root / "src/sword/backend/all_in_one_pipeline.py",
        repo_root / "backend/pipeline/all_in_one_pipelines.py",
    ]
    for p in candidates:
        if p.exists():
            return p
    return Path()

def _start_process(command, cwd: Path, env: dict):
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(cwd),
        env=env,
    )
    logs = []
    for line in process.stdout:
        line = line.strip()
        if line:
            logs.append(line)
    _, stderr = process.communicate()
    return process.returncode, logs, (stderr or "")

@app.post("/start_pipeline/")
def start_pipeline(match_ids: str, background_tasks: BackgroundTasks):
    task_name = f"run_pipeline_{uuid.uuid4().hex[:8]}"

    def pipeline_task():
        task_status[task_name] = {"progress": "Pipeline started...", "finished": False, "error": "", "results": [], "table_data": []}

        try:
            if PROGRESS_PATH.exists():
                PROGRESS_PATH.unlink()
        except Exception:
            pass

        try:
            repo_root = _repo_root()
            env = os.environ.copy()
            env["PYTHONPATH"] = f"{str(repo_root)}{os.pathsep}{env.get('PYTHONPATH','')}"

            # 尝试模块方式
            cmd_mod = [sys.executable, "-m", "src.sword.backend.all_in_one_pipeline", "--match-ids", match_ids]
            rc, logs, stderr = _start_process(cmd_mod, cwd=repo_root, env=env)

            # 若模块失败，回退到脚本路径（兼容 backend/pipeline/all_in_one_pipelines.py）
            if rc != 0:
                script_path = _resolve_pipeline_script(repo_root)
                if script_path.exists():
                    logs.append(f"Fallback to script path: {script_path.name}")
                    cmd_script = [sys.executable, str(script_path), "--match-ids", match_ids]
                    rc, logs2, stderr2 = _start_process(cmd_script, cwd=repo_root, env=env)
                    logs += logs2
                    stderr = stderr2

            # 更新阶段进度
            try:
                if PROGRESS_PATH.exists():
                    with open(PROGRESS_PATH, "r", encoding="utf-8") as f:
                        prog = json.load(f)
                    task_status[task_name]["table_data"] = prog.get("table_data", [])
            except Exception:
                pass

            if rc == 0:
                task_status[task_name]["progress"] = "Pipeline completed successfully!"
                try:
                    with open(RESULT_PATH, "r", encoding="utf-8") as f:
                        pipeline_data = json.load(f)
                except Exception as e:
                    pipeline_data = {"error": f"Failed to load results: {e}", "table_data": []}
                task_status[task_name]["finished"] = True
                task_status[task_name]["results"] = logs + ([stderr] if stderr else [])
                task_status[task_name]["table_data"] = pipeline_data.get("table_data", [])
            else:
                task_status[task_name]["progress"] = "Pipeline failed."
                err_msg = f"Pipeline exited with code {rc}."
                if stderr:
                    err_msg += f" Stderr: {stderr.strip()}"
                task_status[task_name]["error"] = err_msg
                task_status[task_name]["results"] = logs + ([stderr] if stderr else [])
        except Exception as e:
            task_status[task_name]["progress"] = "Pipeline encountered an exception."
            task_status[task_name]["error"] = str(e)

    background_tasks.add_task(pipeline_task)
    return {"message": "Pipeline started successfully.", "task_name": task_name}

@app.get("/task_status/{task_name}")
def get_task_status(task_name: str):
    return task_status.get(task_name, {"error": "Task not found", "progress": "", "finished": False})

@app.get("/paths")
def get_paths():
    return summary()