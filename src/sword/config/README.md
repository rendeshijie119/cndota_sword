# Sword config

Purpose:
- Holds static, human-maintained configuration files that are not downloaded (e.g., map meta, camp boxes).
- Stable paths, versioned in repo.

Contents:
- `camp_boxes_128.json`: Neutral camp spawn boxes for v9 grid (0..128, using x/y-64), side-specific (`radiant`/`dire`).

Override:
- Set `CAMP_BOXES_FILE=/absolute/path/to/camp_boxes_128.json` in environment to use a custom file.

Notes:
- Coordinates must match the ward grid mode (v9: x/y-64 -> 0..128).
- Each box requires `min_x < max_x` and `min_y < max_y`.
- If you see “Camp boxes file not found or empty”, verify the file exists at `src/sword/config/camp_boxes_128.json`, JSON is valid, and boxes are within 0..128.