import json
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

try:
    import torch

    with_torch = True
except ImportError:
    with_torch = False


class DSCamera(object):
    """DSCamera class.
    V. Usenko, N. Demmel, and D. Cremers, "The Double Sphere Camera Model",
    Proc. of the Int. Conference on 3D Vision (3DV), 2018.
    """

    def __init__(
        self,
        json_filename: str = "",
        img_size: Tuple[int, int] = (0, 0),
        intrinsic: Optional[Dict[str, float]] = None,
        fov: float = 180,
    ):
        if json_filename != "":
            # Load data from json file
            with open(json_filename, "r") as f:
                data = json.load(f)
            cam_calib_data = list(data.values())[0]
            intrinsic = cam_calib_data["intrinsics"][0]["intrinsics"]
            _img_size = cam_calib_data["resolution"][0]  # [w, h]
            img_size = (_img_size[1], _img_size[0])  # from [w, h] to [h, w]
            camera_type = cam_calib_data["intrinsics"][0]["camera_type"]
            assert camera_type == "ds", "camera type should be ds"
        assert intrinsic is not None, "Please input json file or parameters."

        # Fisheye camera parameters
        self.h, self.w = img_size
        self.fx = intrinsic["fx"]
        self.fy = intrinsic["fy"]
        self.cx = intrinsic["cx"]
        self.cy = intrinsic["cy"]
        self.xi = intrinsic["xi"]
        self.alpha = intrinsic["alpha"]
        self.fov = fov
        fov_rad = self.fov / 180 * np.pi
        self.fov_cos = np.cos(fov_rad / 2)
        self.intrinsic_keys = ["fx", "fy", "cx", "cy", "xi", "alpha"]

        # Valid mask for fisheye image
        self._valid_mask = None

    @property
    def img_size(self) -> Tuple[int, int]:
        return self.h, self.w

    @img_size.setter
    def img_size(self, img_size: Tuple[int, int]):
        self.h, self.w = map(int, img_size)

    @property
    def intrinsic(self) -> Dict[str, float]:
        intrinsic = {key: self.__dict__[key] for key in self.intrinsic_keys}
        return intrinsic

    @intrinsic.setter
    def intrinsic(self, intrinsic: Dict[str, float]):
        for key in self.intrinsic_keys:
            self.__dict__[key] = intrinsic[key]

    @property
    def valid_mask(self):
        if self._valid_mask is None:
            # Calculate and cache valid mask
            x = np.arange(self.w)
            y = np.arange(self.h)
            x_grid, y_grid = np.meshgrid(x, y, indexing="xy")
            _, valid_mask = self.cam2world([x_grid, y_grid])
            self._valid_mask = valid_mask

        return self._valid_mask

    def __repr__(self):
        return (
            f"[{self.__class__.__name__}]\n img_size:{self.img_size},fov:{self.fov},\n"
            f" intrinsic:{json.dumps(self.intrinsic, indent=2)}"
        )

    def __hash__(self):
        return hash(self.__repr__())

    def __eq__(self, other):
        return self.__repr__() == other.__repr__()

    def cam2world(self, point2D):
        """cam2world(point2D) projects a 2D point onto the unit sphere.
        point3D coord: x:right direction, y:down direction, z:front direction
        point2D coord: x:row direction, y:col direction (OpenCV image coordinate)
        Parameters
        ----------
        point2D : numpy array or list([u,v])
            array of point in image
        Returns
        -------
        unproj_pts : numpy array
            array of point on unit sphere
        valid_mask : numpy array
            array of valid mask
        """
        # Case: point2D = list([u, v]) or np.array()
        if isinstance(point2D, (list, np.ndarray)):
            u, v = point2D
        # Case: point2D = list([Scalar, Scalar])
        if not hasattr(u, "__len__"):
            u, v = np.array([u]), np.array([v])

        # Decide numpy or torch
        if isinstance(u, np.ndarray):
            xp = np
        else:
            xp = torch

        mx = (u - self.cx) / self.fx
        my = (v - self.cy) / self.fy
        r2 = mx * mx + my * my

        # Check valid area
        s = 1 - (2 * self.alpha - 1) * r2
        valid_mask = s >= 0
        s[~valid_mask] = 0.0
        mz = (1 - self.alpha * self.alpha * r2) / (
            self.alpha * xp.sqrt(s) + 1 - self.alpha
        )

        mz2 = mz * mz
        k1 = mz * self.xi + xp.sqrt(mz2 + (1 - self.xi * self.xi) * r2)
        k2 = mz2 + r2
        k = k1 / k2

        # Unprojected unit vectors
        if xp == np:
            unproj_pts = k[..., np.newaxis] * np.stack([mx, my, mz], axis=-1)
        else:
            unproj_pts = k.unsqueeze(-1) * torch.stack([mx, my, mz], dim=-1)
        unproj_pts[..., 2] -= self.xi

        # Calculate fov
        unprojected_fov_cos = unproj_pts[..., 2]  # unproj_pts @ z_axis
        fov_mask = unprojected_fov_cos >= self.fov_cos
        valid_mask *= fov_mask
        return unproj_pts, valid_mask

    def world2cam(self, point3D):
        """world2cam(point3D) projects a 3D point on to the image.
        point3D coord: x:right direction, y:down direction, z:front direction
        point2D coord: x:row direction, y:col direction (OpenCV image coordinate).
        Parameters
        ----------
        point3D : numpy array or list([x, y, z])
            array of points in camera coordinate
        Returns
        -------
        proj_pts : numpy array
            array of points in image
        valid_mask : numpy array
            array of valid mask
        """
        x, y, z = point3D[..., 0], point3D[..., 1], point3D[..., 2]
        # Decide numpy or torch
        if isinstance(x, np.ndarray):
            xp = np
        else:
            xp = torch

        # Calculate fov
        point3D_fov_cos = point3D[..., 2]  # point3D @ z_axis
        fov_mask = point3D_fov_cos >= self.fov_cos

        # Calculate projection
        x2 = x * x
        y2 = y * y
        z2 = z * z
        d1 = xp.sqrt(x2 + y2 + z2)
        zxi = self.xi * d1 + z
        d2 = xp.sqrt(x2 + y2 + zxi * zxi)

        div = self.alpha * d2 + (1 - self.alpha) * zxi
        u = self.fx * x / div + self.cx
        v = self.fy * y / div + self.cy

        # Projected points on image plane
        if xp == np:
            proj_pts = np.stack([u, v], axis=-1)
        else:
            proj_pts = torch.stack([u, v], dim=-1)

        # Check valid area
        if self.alpha <= 0.5:
            w1 = self.alpha / (1 - self.alpha)
        else:
            w1 = (1 - self.alpha) / self.alpha
        w2 = w1 + self.xi / xp.sqrt(2 * w1 * self.xi + self.xi * self.xi + 1)
        valid_mask = z > -w2 * d1
        valid_mask *= fov_mask

        return proj_pts, valid_mask

    def _warp_img(self, img, img_pts, valid_mask):
        # Remap
        img_pts = img_pts.astype(np.float32)
        out = cv2.remap(
            img, img_pts[..., 0], img_pts[..., 1], cv2.INTER_LINEAR
        )
        out[~valid_mask] = 0.0
        return out

    def to_perspective(self, img, img_size=(640, 640), f=0.25):
        # Generate 3D points
        h, w = img_size
        z = f * min(img_size)
        x = np.arange(w) - w / 2
        y = np.arange(h) - h / 2
        x_grid, y_grid = np.meshgrid(x, y, indexing="xy")
        point3D = np.stack([x_grid, y_grid, np.full_like(x_grid, z)], axis=-1)

        # Project on image plane
        img_pts, valid_mask = self.world2cam(point3D)
        out = self._warp_img(img, img_pts, valid_mask)
        return out

    def to_equirect(self, img, img_size=(320, 640)):
        # Generate 3D points
        h, w = img_size
        phi = -np.pi + (np.arange(w) + 0.5) * 2 * np.pi / w
        theta = -np.pi / 2 + (np.arange(h) + 0.5) * np.pi / h
        phi_xy, theta_xy = np.meshgrid(phi, theta, indexing="xy")

        x = np.sin(phi_xy) * np.cos(theta_xy)
        y = np.sin(theta_xy)
        z = np.cos(phi_xy) * np.cos(theta_xy)
        point3D = np.stack([x, y, z], axis=-1)

        # Project on image plane
        img_pts, valid_mask = self.world2cam(point3D)
        out = self._warp_img(img, img_pts, valid_mask)
        return out

    def to_rectilinear(
        self,
        img,
        img_size=(640, 640),
        fov_deg=120,
        pan=0.0,
        tilt=0.0,
        roll=0.0,
        zoom=1.0,
        undistort_amount=1.0,  # New parameter
    ):
        """
        Converts a fisheye image to a rectilinear (perspective) projection with control over pan, tilt, roll, and zoom.

        Parameters
        ----------
        img : numpy array
            Input fisheye image.
        output_size : tuple of int
            Desired output image size as (height, width).
        fov_deg : float
            Field of view in degrees for the rectilinear projection.
        pan : float
            Pan angle in degrees (rotation around the Y-axis).
        tilt : float
            Tilt angle in degrees (rotation around the X-axis).
        roll : float
            Roll angle in degrees (rotation around the Z-axis).
        zoom : float
            Zoom factor (scaling of the focal length).

        Returns
        -------
        rectilinear_img : numpy array
            Rectilinear projected image.
        """
        fov_deg = np.clip(fov_deg, 1, 179)               # Avoid extreme FOVs
        pan = np.clip(pan, -180, 180)
        tilt = np.clip(tilt, -180, 180)
        roll = np.clip(roll, -180, 180)
        zoom = np.clip(zoom, 1, 10)
        undistort_amount = np.clip(undistort_amount, 0.0, 1.0)

        height, width = img_size
        max_fov_deg = 180  # Limiting FOV to prevent extreme distortion
        fov_deg = min(fov_deg, max_fov_deg)
        fov_rad = np.deg2rad(fov_deg / 2)

        # Compute focal length based on desired FOV and zoom
        f = (width / 2) / np.tan(fov_rad)
        f *= zoom  # Apply zoom

        # Generate grid for output image
        u = np.linspace(-width / 2, width / 2 - 1, width)
        v = np.linspace(-height / 2, height / 2 - 1, height)
        u_grid, v_grid = np.meshgrid(u, v, indexing="xy")

        # Normalize coordinates
        x = u_grid / f
        y = v_grid / f
        z = np.ones_like(x)

        # Prepare 3D points in rectilinear projection
        point3D = np.stack((x, y, z), axis=-1)  # Shape: (H, W, 3)

        # Convert angles from degrees to radians and scale by undistort_amount
        pan_rad = np.deg2rad(pan) * undistort_amount
        tilt_rad = np.deg2rad(tilt) * undistort_amount
        roll_rad = np.deg2rad(roll) * undistort_amount

        # Construct rotation matrices
        # Rotation around Y-axis (Pan)
        R_pan = np.array(
            [
                [np.cos(pan_rad), 0, np.sin(pan_rad)],
                [0, 1, 0],
                [-np.sin(pan_rad), 0, np.cos(pan_rad)],
            ]
        )

        # Rotation around X-axis (Tilt)
        R_tilt = np.array(
            [
                [1, 0, 0],
                [0, np.cos(tilt_rad), -np.sin(tilt_rad)],
                [0, np.sin(tilt_rad), np.cos(tilt_rad)],
            ]
        )

        # Rotation around Z-axis (Roll)
        R_roll = np.array(
            [
                [np.cos(roll_rad), -np.sin(roll_rad), 0],
                [np.sin(roll_rad), np.cos(roll_rad), 0],
                [0, 0, 1],
            ]
        )

        # Combined rotation matrix: R = R_roll * R_tilt * R_pan
        R = R_roll @ R_tilt @ R_pan

        # Apply rotation to all 3D points
        # Reshape point3D to (H*W, 3) for matrix multiplication
        H, W, _ = point3D.shape
        points = point3D.reshape(-1, 3).T  # Shape: (3, H*W)
        rotated_points = R @ points  # Shape: (3, H*W)
        rotated_points = rotated_points.T.reshape(H, W, 3)  # Shape: (H, W, 3)

        # Project rotated 3D points to fisheye image plane
        img_pts, valid_mask = self.world2cam(rotated_points)  # Shape: (H, W, 2)

        # Handle points that are out of bounds
        img_pts[..., 0] = np.clip(img_pts[..., 0], 0, self.w - 1)
        img_pts[..., 1] = np.clip(img_pts[..., 1], 0, self.h - 1)

        if undistort_amount <= 0.0:
            # No undistortion; return original fisheye image resized to output_size
            original_img_resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
            return original_img_resized

        elif undistort_amount >= 1.0:
            # Full undistortion; return fully rectilinear image
            rectilinear_img_full = self._warp_img(img, img_pts, valid_mask)
            return rectilinear_img_full

        else:
            # Partial undistortion
            # Generate rectilinear image with scaled rotation
            rectilinear_img_full = self._warp_img(img, img_pts, valid_mask)

            # To blend partial undistortion, we need to compute an intermediate projection
            # Instead of blending images, we adjust the transformation parameters

            # However, since we've already scaled the rotation angles, the current
            # rectilinear_img_full already represents the partial undistortion.

            # Thus, return the partially undistorted image
            return rectilinear_img_full
            
    def to_mobius(
        self,
        img,
        img_size: Tuple[int, int] = (640, 640),
        lambda_deg: float = 0.0,
        phi_deg: float = 0.0,
        s: float = 1.0,
    ):
        """
        Converts a fisheye image to a Möbius projected image.

        Parameters
        ----------
        img : numpy array
            Input fisheye image.
        img_size : tuple of int
            Desired output image size as (height, width).
        lambda_deg : float
            Rotation angle λ in degrees.
        phi_deg : float
            Rotation angle φ in degrees.
        s : float
            Scaling factor for the complex plane.

        Returns
        -------
        mobius_img : numpy array
            Möbius projected image.
        """
        height, width = img_size

        # Convert angles from degrees to radians and invert for rotation
        lambda_rad = np.deg2rad(-lambda_deg)
        phi_rad = np.deg2rad(-phi_deg)

        # Generate grid for output image
        u = np.linspace(-width / 2, width / 2 - 1, width)
        v = np.linspace(-height / 2, height / 2 - 1, height)
        u_grid, v_grid = np.meshgrid(u, v, indexing="xy")

        # Normalize coordinates (Assume z=1 for all points on the projection plane)
        x = u_grid
        y = v_grid
        z = np.ones_like(x)

        # Step 1: Rotation by -lambda on xz plane
        cos_lambda = np.cos(lambda_rad)
        sin_lambda = np.sin(lambda_rad)
        x1 = cos_lambda * x - sin_lambda * z
        y1 = y
        z1 = sin_lambda * x + cos_lambda * z

        # Step 2: Rotation by -phi on yz plane
        cos_phi = np.cos(phi_rad)
        sin_phi = np.sin(phi_rad)
        x2 = x1
        y2 = cos_phi * y1 - sin_phi * z1
        z2 = sin_phi * y1 + cos_phi * z1

        # Step 3: Stereographic projection
        denominator = 1 - z2
        # To avoid division by zero, set denominator to a small number where it's zero
        denominator = np.where(denominator == 0, 1e-8, denominator)
        u_proj = 2 * x2 / denominator
        v_proj = 2 * y2 / denominator

        # Step 4: Convert Cartesian to polar coordinates
        rho = np.sqrt(u_proj**2 + v_proj**2)
        theta = np.arctan2(v_proj, u_proj)

        # Step 5: Scale the complex plane
        rho_scaled = rho * s

        # Step 6: Convert back from polar to Cartesian coordinates
        u_scaled = -rho_scaled * np.sin(theta)
        v_scaled = rho_scaled * np.cos(theta)

        # Step 7: Map back to the unit sphere
        denom = u_scaled**2 + v_scaled**2 + 4
        x3 = 4 * u_scaled / denom
        y3 = 4 * v_scaled / denom
        z3 = (u_scaled**2 + v_scaled**2 - 4) / denom

        # Step 8: Perspective projection
        # Avoid division by zero by setting z3 to a small negative number where z3 >=0
        z3_safe = np.where(z3 >= 0, -1e-8, z3)
        xt = x3 / (-z3_safe)
        yt = y3 / (-z3_safe)
        zt = z3

        # Prepare points for projection: (xt, yt, zt)
        point3D = np.stack([xt, yt, zt], axis=-1)  # Shape: (H, W, 3)

        # Project 3D points to fisheye image plane using world2cam
        img_pts, valid_mask = self.world2cam(point3D)  # Shape: (H, W, 2)

        # Handle points that are out of bounds by clipping
        img_pts[..., 0] = np.clip(img_pts[..., 0], 0, self.w - 1)
        img_pts[..., 1] = np.clip(img_pts[..., 1], 0, self.h - 1)

        # Use the class's _warp_img method for remapping
        mobius_img = self._warp_img(img, img_pts, valid_mask)

        return mobius_img

    def to_panini(self,
        img,
        img_size: Tuple[int, int] = (640, 640),
        d: float = 1.0,
        fov_deg: float = 160,
        pan: float = 0.0,
        tilt: float = 0.0,
        s: float = 1.0,
    ):
        """
        Converts a fisheye image to a Panini projection.

        Parameters
        ----------
        img : numpy array
            Input fisheye image.
        img_size : tuple of int
            Desired output image size as (height, width).
        d : float
            Distance parameter controlling the projection scale (default 1.0).
        fov_deg : float
            Horizontal Field of View in degrees for the Panini projection.
        pan : float
            Pan angle in degrees (rotation around the y-axis).
        tilt : float
            Tilt angle in degrees (rotation around the x-axis).
        s : float
            Compression factor for vertical stretching/compression.

        Returns
        -------
        panini_img : numpy array
            Panini projected image.
        """
        height, width = img_size
        max_fov_deg = 160  # Limiting FOV to prevent extreme distortion
        fov_deg = min(fov_deg, max_fov_deg)
        fov_rad = np.deg2rad(fov_deg)
        half_fov = fov_rad / 2

        # Compute focal length based on desired FOV
        f = (width / 2) / np.tan(half_fov)

        # Generate grid for output image
        u = np.linspace(-width / 2, width / 2 - 1, width)
        v = np.linspace(-height / 2, height / 2 - 1, height)
        u_grid, v_grid = np.meshgrid(u, v, indexing="xy")

        # Compute theta and phi for each pixel
        theta = np.arctan((u_grid) / (d * f))
        phi = np.arctan(v_grid / (f * np.cos(theta)))

        # Adjust theta and phi for pan and tilt
        pan_rad = np.deg2rad(pan)
        tilt_rad = np.deg2rad(tilt)
        theta += pan_rad
        phi *= s  # Apply compression/stretching factor

        # Convert spherical coordinates to 3D Cartesian coordinates
        x = np.cos(phi) * np.sin(theta)
        y = np.sin(phi)
        z = np.cos(phi) * np.cos(theta)

        # Apply tilt rotation (rotation around x-axis)
        if tilt != 0.0:
            cos_tilt = np.cos(tilt_rad)
            sin_tilt = np.sin(tilt_rad)
            y_rot = y * cos_tilt - z * sin_tilt
            z_rot = y * sin_tilt + z * cos_tilt
            y = y_rot
            z = z_rot

        # Apply pan rotation (rotation around y-axis)
        if pan != 0.0:
            cos_pan = np.cos(pan_rad)
            sin_pan = np.sin(pan_rad)
            x_rot = x * cos_pan + z * sin_pan
            z_rot = -x * sin_pan + z * cos_pan
            x = x_rot
            z = z_rot

        # Prepare 3D points
        point3D = np.stack([x, y, z], axis=-1)  # Shape: (H, W, 3)

        # Project 3D points to fisheye image plane using world2cam
        img_pts, valid_mask = self.world2cam(point3D)  # Shape: (H, W, 2)

        # Handle points that are out of bounds by setting to valid_mask
        img_pts[..., 0] = np.clip(img_pts[..., 0], 0, self.w - 1)
        img_pts[..., 1] = np.clip(img_pts[..., 1], 0, self.h - 1)

        # Use the class's _warp_img method for remapping
        panini_img = self._warp_img(img, img_pts, valid_mask)

        return panini_img

    def to_general_panini(
        self,
        img,
        img_size=(640, 640),
        Cmpr=100,
        Tops=0,
        Bots=0,
        hfov_deg: Optional[float] = None,
        vfov_deg: Optional[float] = 120,
        pan=0.0,
        tilt=0.0,
        roll=0.0,
        zoom=1.0,
    ):
        """
        Converts a fisheye image to a general Panini projection with control over pan, tilt, roll, and zoom.

        Parameters
        ----------
        img : numpy array
            Input fisheye image.
        img_size : tuple of int
            Desired output image size as (height, width).
        Cmpr : float
            Compression parameter (0 to 150).
        Tops : float
            Top vertical squeeze parameter (-100 to 100).
        Bots : float
            Bottom vertical squeeze parameter (-100 to 100).
        hfov_deg : float, optional
            Horizontal field of view in degrees. If not provided, it is calculated based on Cmpr.
        vfov_deg : float, optional
            Vertical field of view in degrees. Default is 120.
        pan : float, optional
            Pan angle in degrees (rotation around the Y-axis). Default is 0.0.
        tilt : float, optional
            Tilt angle in degrees (rotation around the X-axis). Default is 0.0.
        roll : float, optional
            Roll angle in degrees (rotation around the Z-axis). Default is 0.0.
        zoom : float, optional
            Zoom factor (scaling of the compression parameter). Default is 1.0.

        Returns
        -------
        panini_img : numpy array
            Panini projected image with controlled orientation and zoom.
        """
        height, width = img_size

        # Validate and clamp parameters
        Cmpr = np.clip(Cmpr, 0, 150)
        Tops = np.clip(Tops, -100, 100)
        Bots = np.clip(Bots, -100, 100)
        if vfov_deg is not None:
            vfov_deg = np.clip(vfov_deg, 1, 179)  # Avoid extreme vertical FOVs

        # Apply zoom to Cmpr
        Cmpr = Cmpr * zoom  # Apply zoom to Cmpr
        # No clamping to allow more flexibility
        # Cmpr = np.clip(Cmpr, 0, 300)

        # Calculate the horizontal field of view based on Cmpr if not provided
        if hfov_deg is None:
            # Example mapping: Adjust based on Cmpr
            # You can customize this mapping based on Hugin's behavior
            if Cmpr < 100:
                hfov_deg_calc = 160 - (160 - 120) * (100 - Cmpr) / 100  # From 160 to 120 as Cmpr decreases
            elif Cmpr <= 200:
                hfov_deg_calc = 120 - (120 - 80) * (Cmpr - 100) / 100  # From 120 to 80 as Cmpr increases
            else:
                hfov_deg_calc = 80 - (80 - 60) * (Cmpr - 200) / 100  # From 80 to 60 as Cmpr increases further
            hfov_deg_calc = np.clip(hfov_deg_calc, 60, 160)  # Extend the range as needed
        else:
            hfov_deg_calc = hfov_deg  # Use provided hfov_deg

        # Use calculated hfov_deg
        hfov_deg_final = hfov_deg_calc

        # Compute focal length based on desired horizontal FOV
        hfov_rad = np.deg2rad(hfov_deg_final / 2)
        f = (width / 2) / np.tan(hfov_rad)

        # Generate grid for Panini projection
        u = np.linspace(-width / 2, width / 2 - 1, width)
        v = np.linspace(-height / 2, height / 2 - 1, height)
        u_grid, v_grid = np.meshgrid(u, v, indexing="xy")

        # Normalize horizontal coordinates
        x = u_grid / f
        y = v_grid / f

        # Calculate the angle theta for each pixel (distance from optical axis)
        theta = np.arctan(np.sqrt(x**2 + y**2))

        # Calculate phi as the azimuthal angle
        phi = np.arctan2(x, 1)  # Assuming cylindrical projection initially

        # Adjust theta based on compression
        # Map Cmpr=0 to no scaling, Cmpr=100 to standard Panini, Cmpr=150 to orthographic
        # Using a smooth scaling function for theta
        scaling_factor = 1 + (Cmpr - 100) / 100  # Example scaling
        theta_scaled = theta * scaling_factor

        # Apply vertical squeeze based on Tops and Bots
        # Normalize v_grid to range [-1, 1]
        v_normalized = v_grid / (height / 2)
        # Compute squeeze factors
        squeeze_top = 1 + (Tops / 100) * np.abs(v_normalized)  # Range: 1 to 2 or 0
        squeeze_bot = 1 + (Bots / 100) * np.abs(v_normalized)

        # Apply squeeze factors
        mask_top = v_grid < 0
        mask_bot = v_grid >= 0

        # Apply squeeze factors
        y_scaled = np.where(mask_top, y * squeeze_top, y * squeeze_bot)

        # Reconstruct 3D points with scaled angles and vertical squeeze
        # Using spherical coordinates to convert back to 3D
        # Assuming the Panini projection maps to the unit sphere
        z = 1 / np.cos(theta_scaled)
        x_final = x * z
        y_final = y_scaled * z
        point3D = np.stack((x_final, y_final, z), axis=-1)  # Shape: (H, W, 3)

        # Normalize the 3D points to lie on the unit sphere
        norm = np.linalg.norm(point3D, axis=-1, keepdims=True)
        point3D_normalized = point3D / norm

        # Apply rotation based on pan, tilt, and roll
        # Convert angles from degrees to radians
        pan_rad = np.deg2rad(pan)
        tilt_rad = np.deg2rad(tilt)
        roll_rad = np.deg2rad(roll)

        # Construct rotation matrices
        # Rotation around Y-axis (Pan)
        R_pan = np.array(
            [
                [np.cos(pan_rad), 0, np.sin(pan_rad)],
                [0, 1, 0],
                [-np.sin(pan_rad), 0, np.cos(pan_rad)],
            ]
        )

        # Rotation around X-axis (Tilt)
        R_tilt = np.array(
            [
                [1, 0, 0],
                [0, np.cos(tilt_rad), -np.sin(tilt_rad)],
                [0, np.sin(tilt_rad), np.cos(tilt_rad)],
            ]
        )

        # Rotation around Z-axis (Roll)
        R_roll = np.array(
            [
                [np.cos(roll_rad), -np.sin(roll_rad), 0],
                [np.sin(roll_rad), np.cos(roll_rad), 0],
                [0, 0, 1],
            ]
        )

        # Combined rotation matrix: R = R_roll * R_tilt * R_pan
        R = R_roll @ R_tilt @ R_pan

        # Apply rotation to all 3D points
        # Reshape point3D_normalized to (H*W, 3) for matrix multiplication
        H, W, _ = point3D_normalized.shape
        points = point3D_normalized.reshape(-1, 3).T  # Shape: (3, H*W)
        rotated_points = R @ points  # Shape: (3, H*W)
        rotated_points = rotated_points.T.reshape(H, W, 3)  # Shape: (H, W, 3)

        # Project rotated 3D points to fisheye image plane using world2cam
        img_pts, valid_mask = self.world2cam(rotated_points)  # Shape: (H, W, 2)

        # Handle points that are out of bounds
        img_pts[..., 0] = np.clip(img_pts[..., 0], 0, self.w - 1)
        img_pts[..., 1] = np.clip(img_pts[..., 1], 0, self.h - 1)

        # Use the class's _warp_img method for remapping
        panini_img = self._warp_img(img, img_pts, valid_mask)

        return panini_img
    
    def to_stereographic(self, img, img_size=(640, 640), f=1.0):
        """
        Perform stereographic projection of the input image.

        Parameters
        ----------
        img : numpy.ndarray
            Input image to be projected.
        img_size : tuple of int, optional
            Size of the output stereographic image as (height, width).
            Default is (640, 640).
        f : float, optional
            Scaling factor for the stereographic projection.
            Controls the extent of the projection. Default is 1.0.

        Returns
        -------
        out : numpy.ndarray
            The stereographically projected image.
        """
        h, w = img_size

        # Create normalized grid coordinates in the stereographic projection plane
        # x and y range from -f to f
        x = (np.linspace(-1, 1, w) * f)
        y = (np.linspace(-1, 1, h) * f)
        x_grid, y_grid = np.meshgrid(x, y, indexing="xy")

        # Compute radius squared
        r_squared = x_grid**2 + y_grid**2

        # Stereographic projection formulas (inverted projection)
        denominator = 1 + r_squared + 1e-8  # Added epsilon for numerical stability
        X = (2 * x_grid) / denominator
        Y = (2 * y_grid) / denominator
        Z = (1 - r_squared) / denominator

        # Flip Z to invert the projection direction
        # This ensures that points in front of the camera map to the center
        # and points behind the camera map to infinity
        point3D = np.stack([X, Y, Z], axis=-1)

        # Project the 3D points onto the camera image plane
        img_pts, valid_mask = self.world2cam(point3D)

        # Remove points that are behind the camera or outside the field of view
        valid_mask &= (point3D[..., 2] > 0)

        # Warp the input image to the stereographic projection
        out = self._warp_img(img, img_pts, valid_mask)

        return out