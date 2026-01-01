from fastapi import FastAPI, BackgroundTasks
import subprocess
import json
from pathlib import Path

app = FastAPI()

# Global task status tracking dictionary
task_status = {}

@app.post("/start_plot_wards/")
def start_plot_wards(options: dict, background_tasks: BackgroundTasks):
    """
    Start the ward plotting pipeline with dynamic options passed via frontend.
    """
    task_name = f"plot_wards_{options.get('team_name', 'unknown')}_{options.get('mid_account', 'unknown')}"

    def plot_wards_task():
        global task_status
        task_status[task_name] = {"progress": "Plotting started...", "finished": False, "error": "", "results": [], "generated_images": []}
        try:
            # Map options to command-line arguments
            command = [
                "python3", "scripts/plot_mid_hero_wards.py",
                f"--team-name", options["team_name"],
                f"--mid-account", options["mid_account"],
                f"--matches-dir", options["matches_dir"],
                f"--wards-dir", options["wards_dir"],
                f"--hero-map", options["hero_map"],
                f"--map", options["map_image"],
                f"--out", options["out_dir"],
                f"--start", str(options.get("start", 0.0)),
                f"--end", str(options.get("end", 10.0)),
                f"--ward-types", options.get("ward_types", "obs"),
                f"--grid-size", str(options.get("grid_size", 256)),
            ]

            if options.get("flip_y", False):
                command.append("--flip-y")
            if options.get("verbose", False):
                command.append("--verbose")
            if options.get("scale_by_lifetime", False):
                command.append("--scale-by-lifetime")
            if options.get("lifetime_auto_range", False):
                command.append("--lifetime-auto-range")
            if options.get("ward_clock", False):
                command.append("--ward-clock")

            # Lifetime scaling options
            command.extend([
                f"--lifetime-min-mul", str(options.get("lifetime_min_mul", 0.6)),
                f"--lifetime-max-mul", str(options.get("lifetime_max_mul", 2.4)),
                f"--obs-max-lifetime", str(options.get("obs_max_lifetime", 360.0)),
                f"--sen-max-lifetime", str(options.get("sen_max_lifetime", 0.0)),
            ])

            # Run the command asynchronously
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            logs = []
            for line in process.stdout:
                logs.append(line.strip())
                task_status[task_name]["progress"] = line.strip()
            process.communicate()

            if process.returncode == 0:
                task_status[task_name]["progress"] = "Plotting completed successfully!"
                # Assuming images are generated in the out directory specified in options
                generated_dir = Path(options["out_dir"])
                generated_images = list(generated_dir.rglob("*.png"))
                task_status[task_name]["generated_images"] = [str(img) for img in generated_images]
                task_status[task_name]["finished"] = True
            else:
                task_status[task_name]["progress"] = "Plotting failed."
                task_status[task_name]["error"] = f"Plotting exited with code {process.returncode}."
        except Exception as e:
            task_status[task_name]["progress"] = "Plotting encountered an exception."
            task_status[task_name]["error"] = str(e)

    background_tasks.add_task(plot_wards_task)
    return {"message": "Ward plotting started successfully.", "task_name": task_name}


@app.get("/task_status/{task_name}")
def get_task_status(task_name: str):
    """
    Get execution status and logs for the specified task.
    """
    return task_status.get(task_name, {"error": "Task not found", "progress": "", "finished": False})


@app.get("/image/")
async def serve_image(file_path: str):
    """
    Serve generated plot images to the frontend.
    """
    plot_file = Path(file_path)
    if not plot_file.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    return FileResponse(plot_file)