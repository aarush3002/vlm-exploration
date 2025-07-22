#!/usr/bin/env python3
"""
Roam in Habitat WITHOUT starting any joystick‑related nodes.
Everything else (Habitat env‑node, image views, scanners…)
works exactly like roam_with_joy.
"""
import argparse
from src.roamers.joy_habitat_roamer import JoyHabitatRoamer

def main() -> None:
    p = argparse.ArgumentParser()
    # launch file that *doesn't* start joy nodes  ↓
    p.add_argument("--launch-file-path",   default="launch/roam_no_joy.launch")
    p.add_argument("--hab-env-node-path",  default="src/nodes/habitat_env_node_new.py")
    p.add_argument("--hab-env-config-path",default="configs/pointnav_rgbd_roam.yaml")
    p.add_argument("--hab-env-node-name",  default="roamer_env_node")
    p.add_argument("--episode-id",         default="-1")
    p.add_argument("--scene-id",
                   default="data/scene_datasets/habitat-test-scenes/"
                           "skokloster-castle.glb")
    p.add_argument("--video-frame-period", type=int, default=60)
    args = p.parse_args()

    roamer = JoyHabitatRoamer(
        launch_file_path     = args.launch_file_path,
        hab_env_node_path    = args.hab_env_node_path,
        hab_env_config_path  = args.hab_env_config_path,
        hab_env_node_name    = args.hab_env_node_name,
        video_frame_period   = args.video_frame_period,
    )
    roamer.roam_until_shutdown(args.episode_id, args.scene_id)

if __name__ == "__main__":
    main()
