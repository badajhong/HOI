"""RealSense depth camera helper.

This script uses `pyrealsense2` only for camera capture. It reads true Z16
depth frames, queries the device depth scale, and returns depth images in
meters.

Requires: numpy, pyrealsense2. OpenCV is optional and only used for --streaming display.
"""

from typing import Optional

import time

import numpy as np


class RealSenseDepthCamera:
	"""Open a RealSense depth stream through pyrealsense2."""

	def __init__(
		self,
		width: int = 640,
		height: int = 480,
		fps: int = 30,
		output_width: Optional[int] = 80,
		output_height: Optional[int] = 60,
		min_depth_m: float = 0.05,
		max_depth_m: float = 20.0,
		queue_size: int = 1,
		discard_initial_frames: int = 30,
	):
		try:
			import pyrealsense2 as rs
		except Exception as exc:
			raise RuntimeError(
				"pyrealsense2 is required. Run scripts/setup_rsinference.sh and activate the rsinference env."
			) from exc

		self.rs = rs
		self.width = width
		self.height = height
		self.fps = fps
		self.output_width = output_width
		self.output_height = output_height
		self.min_depth_m = min_depth_m
		self.max_depth_m = max_depth_m
		self.queue_size = queue_size
		self.discard_initial_frames = discard_initial_frames
		self.pipeline = rs.pipeline()
		self.config = rs.config()
		self.profile = None
		self.depth_scale = 0.0
		self._resize_rows: Optional[np.ndarray] = None
		self._resize_cols: Optional[np.ndarray] = None

	def __enter__(self):
		self.config.enable_stream(self.rs.stream.depth, self.width, self.height, self.rs.format.z16, self.fps)
		self.profile = self.pipeline.start(self.config)
		depth_sensor = self.profile.get_device().first_depth_sensor()
		self.depth_scale = float(depth_sensor.get_depth_scale())
		try:
			if depth_sensor.supports(self.rs.option.frames_queue_size):
				depth_sensor.set_option(self.rs.option.frames_queue_size, float(self.queue_size))
		except Exception:
			pass
		for _ in range(max(0, self.discard_initial_frames)):
			self.pipeline.wait_for_frames()
		self._configure_resize()
		return self

	def __exit__(self, exc_type, exc, tb):
		self.pipeline.stop()

	def read(self) -> np.ndarray:
		"""Return a float32 depth image in meters."""
		frames = self.pipeline.wait_for_frames()
		depth_frame = frames.get_depth_frame()
		if not depth_frame:
			raise RuntimeError("Failed to read RealSense depth frame")

		depth_raw = np.asanyarray(depth_frame.get_data()).astype(np.uint16, copy=False)
		if self._resize_rows is not None and self._resize_cols is not None:
			depth_raw = depth_raw[self._resize_rows[:, None], self._resize_cols[None, :]]
		depth_m = depth_raw.astype(np.float32) * self.depth_scale
		invalid = ~np.isfinite(depth_m) | (depth_raw == 0) | (depth_raw == 65535)
		if self.min_depth_m > 0.0:
			invalid |= depth_m < self.min_depth_m
		if self.max_depth_m > 0.0:
			invalid |= depth_m > self.max_depth_m
		depth_m[invalid] = 0.0
		return depth_m

	def _configure_resize(self) -> None:
		stream_profile = self.profile.get_stream(self.rs.stream.depth).as_video_stream_profile()
		src_width = int(stream_profile.width())
		src_height = int(stream_profile.height())
		if not self.output_width or not self.output_height or (src_width, src_height) == (
			self.output_width,
			self.output_height,
		):
			return
		self._resize_rows = np.linspace(0, src_height - 1, self.output_height).round().astype(np.int64)
		self._resize_cols = np.linspace(0, src_width - 1, self.output_width).round().astype(np.int64)

	def stream_description(self) -> str:
		if self.profile is None:
			return f"requested_stream={self.width}x{self.height}@{self.fps}fps"
		stream_profile = self.profile.get_stream(self.rs.stream.depth).as_video_stream_profile()
		return (
			f"actual_stream={stream_profile.width()}x{stream_profile.height()}@{stream_profile.fps()}fps "
			f"queue_size={self.queue_size}"
		)


def depth_stats(
	depth_m: np.ndarray,
	min_valid_m: Optional[float],
	max_valid_m: Optional[float],
	low_percentile: float,
	high_percentile: float,
) -> str:
	valid = depth_m[depth_m > 0.0]
	total_valid = int(valid.size)
	if min_valid_m is not None and min_valid_m > 0.0:
		valid = valid[valid >= min_valid_m]
	if max_valid_m is not None and max_valid_m > 0.0:
		valid = valid[valid <= max_valid_m]
	if valid.size == 0:
		return f"dtype:{depth_m.dtype} shape:{depth_m.shape} min_m:0.000 max_m:0.000 valid:0 dropped:{total_valid}"

	lo, hi, median = np.percentile(valid, [low_percentile, high_percentile, 50.0])
	dropped = total_valid - int(valid.size)

	return (
		f"dtype:{depth_m.dtype} shape:{depth_m.shape} "
		f"min_m:{float(lo):.3f} max_m:{float(hi):.3f} "
		f"median_m:{float(median):.3f} mean_m:{float(valid.mean()):.3f} "
		f"valid:{int(valid.size)} dropped:{dropped}"
	)


def depth_preview_u8(depth_m: np.ndarray, max_m: float) -> np.ndarray:
	preview = np.clip(depth_m / max(max_m, 1e-6), 0.0, 1.0)
	preview[depth_m <= 0.0] = 0.0
	return (preview * 255.0).astype(np.uint8)


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description="RealSense pyrealsense2 depth camera demo")
	parser.add_argument("--width", type=int, default=640, help="RealSense depth stream width")
	parser.add_argument("--height", type=int, default=480, help="RealSense depth stream height")
	parser.add_argument("--fps", type=int, default=90, help="RealSense depth stream FPS")
	parser.add_argument("--output-width", type=int, default=80, help="resized output width; use 0 to keep native")
	parser.add_argument("--output-height", type=int, default=60, help="resized output height; use 0 to keep native")
	parser.add_argument("--min-depth-m", type=float, default=0.05, help="minimum valid depth in meters")
	parser.add_argument("--max-depth-m", type=float, default=20.0, help="maximum valid depth in meters")
	parser.add_argument("--queue-size", type=int, default=1, help="RealSense frame queue size for low latency")
	parser.add_argument("--discard-initial-frames", type=int, default=30, help="warmup frames to discard after start")
	parser.add_argument("--stats-period", type=float, default=1.0, help="seconds between live stats prints")
	parser.add_argument("--stats-min-m", type=float, default=0.05, help="ignore smaller values in printed stats; use 0 to disable")
	parser.add_argument("--stats-max-m", type=float, default=20.0, help="ignore larger values in printed stats; use 0 to disable")
	parser.add_argument("--stats-low-percentile", type=float, default=5.0, help="percentile used as printed min_m")
	parser.add_argument("--stats-high-percentile", type=float, default=95.0, help="percentile used as printed max_m")
	parser.add_argument("--streaming", action="store_true", help="show live depth stream in an OpenCV window")
	parser.add_argument("--stream-max-m", type=float, default=20.0, help="OpenCV depth display range in meters")
	args = parser.parse_args()

	output_width = args.output_width if args.output_width > 0 else None
	output_height = args.output_height if args.output_height > 0 else None
	stats_min_m = args.stats_min_m if args.stats_min_m > 0.0 else None
	stats_max_m = args.stats_max_m if args.stats_max_m > 0.0 else None

	try:
		cv2 = None
		if args.streaming:
			import cv2

		with RealSenseDepthCamera(
			width=args.width,
			height=args.height,
			fps=args.fps,
			output_width=output_width,
			output_height=output_height,
			min_depth_m=args.min_depth_m,
			max_depth_m=args.max_depth_m,
			queue_size=args.queue_size,
			discard_initial_frames=args.discard_initial_frames,
		) as cam:
			print(
				f"Reading RealSense depth at {args.width}x{args.height} @ {args.fps}fps, "
				f"depth_scale={cam.depth_scale:.8f} m/unit."
			)
			print(cam.stream_description())
			if output_width and output_height:
				print(f"Output resized with nearest-neighbor to {output_width}x{output_height}. Press Ctrl-C to stop.")
			else:
				print("Output kept at native stream size. Press Ctrl-C to stop.")

			last = time.time()
			frames = 0
			total = 0
			while True:
				depth_m = cam.read()
				frames += 1
				total += 1
				now = time.time()

				if now - last >= args.stats_period:
					stats = (
						f"FPS:{frames / max(now - last, 1e-6):.1f} total:{total} "
						f"{depth_stats(depth_m, stats_min_m, stats_max_m, args.stats_low_percentile, args.stats_high_percentile)}"
					)
					print(stats, flush=True)
					frames = 0
					last = now

				if args.streaming and cv2 is not None:
					depth_u8 = depth_preview_u8(depth_m, args.stream_max_m)
					depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)
					cv2.imshow("RealSense Depth", depth_color)
					if cv2.waitKey(1) & 0xFF == ord("q"):
						break

	except KeyboardInterrupt:
		print("\nInterrupted by user")
	except Exception as exc:
		print("Demo failed:", exc)
