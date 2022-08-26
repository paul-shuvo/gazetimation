import mediapipe as mp
import cv2
import numpy as np


class Gazetimation:
    def __init__(
        self,
        face_model_points_3d: np.ndarray = None,
        left_eye_ball_center: np.ndarray = None,
        right_eye_ball_center: np.ndarray = None,
        camera_matrix: np.ndarray = None,
        device: int = 0,
        visualize: bool = True,
    ) -> None:
        """Initialize the Gazetimation object.

        This holds the configurations of the Gazetimation class.

        Args:
            face_model_points_3d (np.ndarray, optional): Predefine 3D reference points for face model. Defaults to None.
            
                .. note::
                    If not provided, it will be assigned the following values. And, the passed values should conform to the same facial points.
                
                    .. code-block:: python

                        self._face_model_points_3d = np.array(
                            [
                                (0.0, 0.0, 0.0),  # Nose tip
                                (0, -63.6, -12.5),  # Chin
                                (-43.3, 32.7, -26),  # Left eye, left corner
                                (43.3, 32.7, -26),  # Right eye, right corner
                                (-28.9, -28.9, -24.1),  # Left Mouth corner
                                (28.9, -28.9, -24.1),  # Right mouth corner
                            ]
                        )
                
            left_eye_ball_center (np.ndarray, optional): Predefine 3D reference points for left eye ball center. Defaults to None.
            
                .. note::
                    If not provided, it will be assigned the following values. And, the passed values should conform to the same facial points.
                
                    .. code-block:: python
                        
                        self._left_eye_ball_center = np.array([[29.05], [32.7], [-39.5]])
            
            right_eye_ball_center (np.ndarray, optional): Predefine 3D reference points for right eye ball center. Defaults to None.
            
                .. note::
                    If not provided, it will be assigned the following values. And, the passed values should conform to the same facial points.
                
                    .. code-block:: python
                        
                        self._right_eye_ball_center = np.array([[-29.05], [32.7], [-39.5]])
            
            camera_matrix (np.ndarray, optional): Camera matrix. Defaults to None.
            
                .. important::
                    | if not provided, the system tries to calculate the camera matrix using the :py:meth:`find_camera_matrix method <gazetimation.Gazetimation.find_camera_matrix>`.
                    | This calculated camera matrix is estimated from the width and height of the frame, it's not an exact solution.
                        
            device (int, optional): Device index for the video device. Defaults to 0.

                .. attention::
                    if a negative device index is provided, the system tries to find the first available video device index using the :py:meth:`find_device method <gazetimation.Gazetimation.find_device>`. 
                    So, if not sure, pass `device = -1`.

                    .. code-block:: python
                    
                        if device < 0:
                            self._device = self.find_device()
                        else:
                            self._device = device
                        
            
            visualize (bool, optional): If visualize is true then it shows annotated images. Defaults to True.
        """
        if not face_model_points_3d:
            self._face_model_points_3d = np.array(
                [
                    (0.0, 0.0, 0.0),  # Nose tip
                    (0, -63.6, -12.5),  # Chin
                    (-43.3, 32.7, -26),  # Left eye, left corner
                    (43.3, 32.7, -26),  # Right eye, right corner
                    (-28.9, -28.9, -24.1),  # Left Mouth corner
                    (28.9, -28.9, -24.1),  # Right mouth corner
                ]
            )
        else:
            self._face_model_points_3d = face_model_points_3d

        # 3D model eye points
        # The center of the eye ball

        if not left_eye_ball_center:
            self._left_eye_ball_center = np.array([[29.05], [32.7], [-39.5]])
        else:
            self._left_eye_ball_center = left_eye_ball_center

        if not right_eye_ball_center:
            self._right_eye_ball_center = np.array([[-29.05], [32.7], [-39.5]])
        else:
            self._right_eye_ball_center = right_eye_ball_center

        # Nose tip -> 4
        # Chin -> 152
        # Left eye left corner -> 263
        # Right eye right corner -> 33
        # Left Mouth corner -> 287
        # Right mouth corner -> 57
        self._facial_landmark_index = [4, 152, 263, 33, 287, 57]
        self._camera_matrix = camera_matrix
        if device < 0:
            self._device = self.find_device()
        else:
            self._device = device
        self._visualize = visualize

    @property
    def face_model_points_3d(self) -> np.ndarray:
        """Getter method for face_model_points_3d.
        Returns:
            np.ndarray: 3D face model points.
        """
        return self._face_model_points_3d

    @property
    def left_eye_ball_center(self) -> np.ndarray:
        """Getter method for left_eye_ball_center.

        Returns:
            np.ndarray: 3D points for left eye ball center.
        """
        return self._left_eye_ball_center

    @property
    def right_eye_ball_center(self) -> np.ndarray:
        """Getter method for right_eye_ball_center.

        Returns:
            np.ndarray: 3D points for right eye ball center.
        """
        return self._right_eye_ball_center

    @property
    def facial_landmark_index(self) -> list:
        """Getter method for facial_landmark_index.

        Returns:
            list: Required facial landmark indexes.
        """
        return self._facial_landmark_index

    @property
    def camera_matrix(self) -> np.ndarray:
        """Getter method for camera_matrix.

        Returns:
            np.ndarray: The camera matrix.
        """
        return self._camera_matrix

    @property
    def device(self) -> int:
        """Getter method for device.

        Returns:
            int: Index for the video device.
        """
        return self._device

    @property
    def visualize(self) -> bool:
        """Getter method for visualize.

        Returns:
            bool: Whether to show annotated images.
        """
        return self._visualize

    @face_model_points_3d.setter
    def face_model_points_3d(self, value: np.ndarray):
        """Setter method for face_model_points_3d.

        Args:
            value (np.ndarray): New/updated value.
        """
        self._face_model_points_3d = value

    @left_eye_ball_center.setter
    def left_eye_ball_center(self, value: np.ndarray):
        """Setter method for left_eye_ball_center.

        Args:
            value (np.ndarray): New/updated value.
        """
        self._left_eye_ball_center = value

    @right_eye_ball_center.setter
    def right_eye_ball_center(self, value: np.ndarray):
        """Setter method for right_eye_ball_center.

        Args:
            value (np.ndarray): New/updated value.
        """
        self._right_eye_ball_center = value

    @facial_landmark_index.setter
    def facial_landmark_index(self, value: list):
        """Setter method for facial_landmark_index.

        Args:
            value (list): New/updated value.
        """
        self._facial_landmark_index = value

    @camera_matrix.setter
    def camera_matrix(self, value: np.ndarray):
        """Setter method for camera_matrix.

        Args:
            value (np.ndarray): New/updated value.
        """
        self._camera_matrix = value

    @device.setter
    def device(self, value: int):
        """Setter method for device.

        Args:
            value (int): New/updated value.
        """
        self._device = value

    @visualize.setter
    def visualize(self, value: bool):
        """Setter method for visualize.

        Args:
            value (bool): New/updated value.
        """
        self._visualize = value

    def find_device(self, max_try: int = 10) -> int:
        """Find the video device index.

        It tries to iterate over a number of system device
        and returns the first eligible device.

        Args:
            max_try (int, optional): Max number of devices to try. Defaults to 10.

        Returns:
            int: Index of the video device.
        """
        for device in range(max_try):
            cap = cv2.VideoCapture(device)
            while cap.isOpened():
                success, _ = cap.read()
                if success:
                    return device
        return -1

    def find_camera_matrix(self, frame: np.ndarray) -> np.ndarray:
        """Calculates the camera matrix from image dimensions.

        Args:
            frame (np.ndarray): The image.

        Returns:
            np.ndarray: Camera matrix.
        """
        focal_length = frame.shape[1]
        center = (frame.shape[1] / 2, frame.shape[0] / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
            dtype="double",
        )

        return camera_matrix

    def find_face_num(self, max_try: int = 100, video_path: str = None) -> int:
        """Finds number of faces/people present in the scene

        Args:
            max_try (int, optional): Maximum number of frames to try. Defaults to 100.
            video_path (str, optional): Path to the video file. Defaults to None.

        Returns:
            int: The number of faces/people present in the scene.
        """
        mp_face_detection = mp.solutions.face_detection
        if video_path:
            cap = cv2.VideoCapture(video_path)
        else:
            cap = cv2.VideoCapture(self.device)

        with mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        ) as face_detection:
            for try_ in range(max_try):
                success, frame = cap.read()
                if success:
                    break
                if try_ == max_try - 1:
                    return -1
            # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
            results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Draw face detections of each face.
            if not results.detections:
                return 0
            else:
                return len(results.detections)

    def calculate_head_eye_poses(
        self, frame: np.ndarray, points: object, gaze_distance: int = 10, face_model_points_3d: np.ndarray=None
    ) -> tuple:
        """Calculates the head and eye poses (gaze)

        Args:
            frame (np.ndarray): The image.
            points (object): Holds the facial landmarks points.
            gaze_distance (int, optional): Gaze distance. Defaults to 10.
            face_model_points_3d (np.ndarray, optional): Predefined 3D reference points for face model. Defaults to None.


        Returns:
            tuple: Returns two tuples (left and right eye) containing the pupil location and the projected gaze on the image plane.
        """

        frame_height, frame_width, _ = frame.shape
        
        # If face_model_points_3d is provided
        # update the object variable and use
        # the updated values.
        if face_model_points_3d:
            self.face_model_points_3d = face_model_points_3d
        
        # Mediapipe points are normalized to [-1, 1].
        # Image points holds the landmark points in terms of
        # image coordinates
        image_points = np.array(
            [
                (
                    points.landmark[ind].x * frame_width,
                    points.landmark[ind].y * frame_height,
                )
                for ind in self.facial_landmark_index
            ]
        )

        # 0 is added for each image_points e.g. in (x,y,0) format
        # (stored in image_points_ext) to find the transformation
        # from image to world points
        image_points_ext = np.hstack(
            (image_points, np.zeros((image_points.shape[0], 1)))
        )
        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
        _, rotation_vector, translation_vector = cv2.solvePnP(
            self.face_model_points_3d,
            image_points,
            self.camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        # 2d pupil location
        left_pupil = np.array(
            (
                points.landmark[468].x * frame_width,
                points.landmark[468].y * frame_height,
            )
        )
        right_pupil = np.array(
            (
                points.landmark[473].x * frame_width,
                points.landmark[473].y * frame_height,
            )
        )

        # Transformation between image point to world point
        success, transformation, _ = cv2.estimateAffine3D(
            image_points_ext, self.face_model_points_3d
        )

        # if estimateAffine3D was successful
        # project pupil image point into 3d world point
        if success:
            pupil_world_cord = (
                transformation @ np.array([[left_pupil[0], left_pupil[1], 0, 1]]).T,
                transformation @ np.array([[right_pupil[0], right_pupil[1], 0, 1]]).T,
            )

            # 3D gaze point (10 is arbitrary value denoting gaze distance)
            gaze_point_3D = [
                self.left_eye_ball_center
                + (pupil_world_cord[0] - self.left_eye_ball_center) * gaze_distance,
                self.right_eye_ball_center
                + (pupil_world_cord[1] - self.right_eye_ball_center) * gaze_distance,
            ]

            # Project a 3D gaze direction onto the image plane.
            gaze_direction_left_eye, _ = cv2.projectPoints(
                (
                    int(gaze_point_3D[0][0]),
                    int(gaze_point_3D[0][1]),
                    int(gaze_point_3D[0][2]),
                ),
                rotation_vector,
                translation_vector,
                self.camera_matrix,
                dist_coeffs,
            )
            gaze_direction_right_eye, _ = cv2.projectPoints(
                (
                    int(gaze_point_3D[1][0]),
                    int(gaze_point_3D[1][1]),
                    int(gaze_point_3D[1][2]),
                ),
                rotation_vector,
                translation_vector,
                self.camera_matrix,
                dist_coeffs,
            )
            # project 3D head pose into the image plane
            head_pose_left, _ = cv2.projectPoints(
                (int(pupil_world_cord[0][0]), int(pupil_world_cord[0][1]), int(40)),
                rotation_vector,
                translation_vector,
                self.camera_matrix,
                dist_coeffs,
            )

            head_pose_right, _ = cv2.projectPoints(
                (int(pupil_world_cord[1][0]), int(pupil_world_cord[1][1]), int(40)),
                rotation_vector,
                translation_vector,
                self.camera_matrix,
                dist_coeffs,
            )
            # correct gaze for head rotation
            gaze_left_eye = (
                left_pupil
                + (gaze_direction_left_eye[0][0] - left_pupil)
                - (head_pose_left[0][0] - left_pupil)
            )

            gaze_right_eye = (
                right_pupil
                + (gaze_direction_right_eye[0][0] - right_pupil)
                - (head_pose_right[0][0] - right_pupil)
            )

            return (left_pupil, right_pupil), (gaze_left_eye, gaze_right_eye)

    def smoothing(
        self,
        smoothing_weight: str,
        smoothing_frame_range: int,
        left_pupil: np.ndarray,
        right_pupil: np.ndarray,
        gaze_left_eye: np.ndarray,
        gaze_right_eye: np.ndarray,
    ) -> tuple:
        """Smoothing is performed so the result doesn't have abrupt changes.

        Args:
            smoothing_weight (str): Type of smoothing.
            smoothing_frame_range (int): Number of frame to consider to perform smoothing.
            left_pupil (np.ndarray): Position of the left pupil.
            right_pupil (np.ndarray): Position of the right pupil.
            gaze_left_eye (np.ndarray): Position of the estimated gaze of left eye.
            gaze_right_eye (np.ndarray): Position of the estimated gaze of right eye.

        Returns:
            tuple : Smoothed position of left_pupil, right_pupil, gaze_left_eye, gaze_right_eye
        """        
        if not self.gaze_data:
            self.gaze_data = {
                'left_pupil': np.tile(np.array(left_pupil), (smoothing_frame_range, 1)),
                'right_pupil': np.tile(
                    np.array(right_pupil), (smoothing_frame_range, 1)
                ),
                'gaze_left_eye': np.tile(
                    np.array(gaze_left_eye), (smoothing_frame_range, 1)
                ),
                'gaze_right_eye': np.tile(
                    np.array(gaze_right_eye), (smoothing_frame_range, 1)
                ),
            }
            if smoothing_weight == 'linear':
                self.weight = np.arange(1, smoothing_frame_range + 1)
                self.weight = self.weight / np.sum(self.weight)
            elif smoothing_weight == 'logarithmic':
                self.weight = np.logspace(0, 2.0, num=smoothing_frame_range)
                self.weight = self.weight / np.sum(self.weight)

            return left_pupil, right_pupil, gaze_left_eye, gaze_right_eye
        self.gaze_data['left_pupil'][1:] = self.gaze_data['left_pupil'][:-1]
        self.gaze_data['left_pupil'][0] = left_pupil

        self.gaze_data['right_pupil'][1:] = self.gaze_data['right_pupil'][:-1]
        self.gaze_data["right_pupil"][0] = right_pupil

        self.gaze_data['gaze_left_eye'][1:] = self.gaze_data['gaze_left_eye'][:-1]
        self.gaze_data['gaze_left_eye'][0] = gaze_left_eye

        self.gaze_data['gaze_right_eye'][1:] = self.gaze_data['gaze_right_eye'][:-1]
        self.gaze_data['gaze_right_eye'][0] = gaze_right_eye

        if smoothing_weight == "uniform":
            left_pupil_, right_pupil_, gaze_left_eye_, gaze_right_eye_ = (
                np.mean(self.gaze_data['left_pupil'], axis=0),
                np.mean(self.gaze_data['right_pupil'], axis=0),
                np.mean(self.gaze_data['gaze_left_eye'], axis=0),
                np.mean(self.gaze_data['gaze_right_eye'], axis=0),
            )
        elif smoothing_weight == 'linear' or smoothing_weight == 'logarithmic':
            left_pupil_, right_pupil_, gaze_left_eye_, gaze_right_eye_ = (
                np.sum(np.einsum('ij, i -> ij', self.gaze_data['left_pupil'], self.weight), axis=0),
                np.sum(np.einsum('ij, i -> ij', self.gaze_data['right_pupil'], self.weight), axis=0),
                np.sum(np.einsum('ij, i -> ij', self.gaze_data['gaze_left_eye'], self.weight), axis=0),
                np.sum(np.einsum('ij, i -> ij', self.gaze_data['gaze_right_eye'], self.weight), axis=0),
            )

        return left_pupil_, right_pupil_, gaze_left_eye_, gaze_right_eye_

    def run(
        self,
        max_num_faces: int = 1,
        video_path: str = None,
        smoothing: bool =True,
        smoothing_frame_range: int=8,
        smoothing_weight="uniform",
        custom_smoothing_func =None,
        video_output_path: str = None
    ):
        """Runs the solution

        Args:
            max_num_faces (int, optional): Maximum number of face(s)/people present in the scene. Defaults to 1.
            video_path (str, optional): Path to the video. Defaults to None.
            smoothing (bool, optional): If smoothing should be performed. Defaults to True.
            smoothing_frame_range (int, optional): Number of frame to consider to perform smoothing.. Defaults to 8.
            smoothing_weight (str, optional): Type of weighting scheme ("uniform", "linear", "logarithmic"). Defaults to "uniform".
            custom_smoothing_func (function, optional): Custom smoothing function. Defaults to None.
            video_output_path (str, optional): Output path and format for output video.
        """
        if video_output_path:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out = cv2.VideoWriter(video_output_path, fourcc, 20.0, (640,480))
            
        if smoothing:
            self.gaze_data = None
            if not custom_smoothing_func:
                smoothing_func = self.smoothing
            else:
                smoothing_func = custom_smoothing_func
        mp_face_mesh = mp.solutions.face_mesh  # initialize the face mesh model
        assert self.device >= 0
        if video_path:
            cap = cv2.VideoCapture(video_path)
        else:
            cap = cv2.VideoCapture(self.device)  # chose camera index (try 1, 2, 3)
            
        with mp_face_mesh.FaceMesh(
            max_num_faces=max_num_faces,  # number of faces to track in each frame
            refine_landmarks=True,  # includes iris landmarks in the face mesh model
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as face_mesh:
            while cap.isOpened():
                success, frame = cap.read()
                if self.camera_matrix is None:
                    self.camera_matrix = self.find_camera_matrix(frame)
                if not success:  # no frame input
                    print("Ignoring empty camera frame.")
                    continue
                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                frame.flags.writeable = False
                frame = cv2.cvtColor(
                    frame, cv2.COLOR_BGR2RGB
                )  # frame to RGB for the face-mesh model
                results = face_mesh.process(frame)
                frame = cv2.cvtColor(
                    frame, cv2.COLOR_RGB2BGR
                )  # frame back to BGR for OpenCV

                if results.multi_face_landmarks:
                    for face_num in range(max_num_faces):
                        try:
                            (left_pupil, right_pupil), (
                                gaze_left_eye,
                                gaze_right_eye,
                            ) = self.calculate_head_eye_poses(
                                frame, results.multi_face_landmarks[face_num]
                            )  # gaze estimation
                            if smoothing:
                                (
                                    left_pupil,
                                    right_pupil,
                                    gaze_left_eye,
                                    gaze_right_eye,
                                ) = smoothing_func(
                                    smoothing_weight,
                                    smoothing_frame_range,
                                    left_pupil,
                                    right_pupil,
                                    gaze_left_eye,
                                    gaze_right_eye,
                                )
                        except TypeError as error:
                            print(f"TypeError: {error}")
                            continue
                        if self.visualize:
                            self.draw(frame, left_pupil, gaze_left_eye)
                            self.draw(frame, right_pupil, gaze_right_eye)
                        if video_output_path:
                            out.write(frame)
                cv2.imshow("output window", frame)
                if cv2.waitKey(2) & 0xFF == 27:
                    break
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    def draw(self, frame: np.ndarray, pupil: np.ndarray, gaze: np.ndarray):
        """Draws the gaze direction onto the frame

        Args:
            frame (np.ndarray): The image.
            pupil (np.ndarray): 2D pupil location on the image.
            gaze (np.ndarray): Gaze direction.
        """        
        # Draw gaze line into screen
        p1 = (int(pupil[0]), int(pupil[1]))
        p2 = (int(gaze[0]), int(gaze[1]))
        cv2.line(frame, p1, p2, (0, 0, 255), 2)

        v1, v2 = self.calculate_arrowhead(p1, p2)
        cv2.circle(frame, center=p1, radius=6, color=(173, 68, 142), thickness=2)
        cv2.line(frame, v1, p2, (0, 255, 0), 2)
        cv2.line(frame, v2, p2, (0, 255, 0), 2)

    def calculate_arrowhead(
        self,
        start_coordinate: tuple,
        end_coordinate: tuple,
        arrow_length: int = 15,
        arrow_angle: int = 70,
    ) -> tuple:
        """Calculate the lines for arrowhead.

        For a given line, it calculates the arrowhead
        from the end (tip) of the line.

        Args:
            start_coordinate (tuple): Start point of the line.
            end_coordinate (tuple): End point of the line.
            arrow_length (int, optional): Length of the arrowhead lines. Defaults to 15.
            arrow_angle (int, optional): Angle (degree) of the arrowhead lines with the arrow-line. Defaults to 70.

        Returns:
            tuple: Endpoints in the image of the two arrowhead lines.
        """
        angle = (
            np.arctan2(
                end_coordinate[1] - start_coordinate[1],
                end_coordinate[0] - start_coordinate[0],
            )
            + np.pi
        )
        arrow_length = 15

        x1 = int(end_coordinate[0] + arrow_length * np.cos(angle - arrow_angle))
        y1 = int(end_coordinate[1] + arrow_length * np.sin(angle - arrow_angle))
        x2 = int(end_coordinate[0] + arrow_length * np.cos(angle + arrow_angle))
        y2 = int(end_coordinate[1] + arrow_length * np.sin(angle + arrow_angle))

        return (x1, y1), (x2, y2)


# g = Gazetimation(device=1)
# g.camera_matrix = np.array(
#     [
#         [1394.6027293299926, 0, 995.588675691456],
#         [0, 1394.6027293299926, 599.3212928484164],
#         [0, 0, 1],
#     ]
# )
# # print(g.get_face_num())
# g.run(smoothing_weight='linear', smoothing_frame_range=8)
