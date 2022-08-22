import mediapipe as mp
import cv2
import numpy as np
from helpers import relative, relativeT, calc_vertexes


class Gazetimation:
    def __init__(
        self,
        face_model_points_3d: np.ndarray = None,
        left_eye_ball_center: np.ndarray = None,
        right_eye_ball_center: np.ndarray = None,
        camera_matrix: np.ndarray = None,
        device: int = -1,
        visualize: bool = True,
    ) -> None:

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
    def face_model_points_3d(self):
        return self._face_model_points_3d
    
    @property
    def left_eye_ball_center(self):
        return self._left_eye_ball_center
    
    @property
    def right_eye_ball_center(self):
        return self._right_eye_ball_center
    
    @property
    def facial_landmark_index(self):
        return self._facial_landmark_index
    
    @property
    def camera_matrix(self):
        return self._camera_matrix
    
    @property
    def device(self):
        return self._device
    
    @property
    def visualize(self):
        return self._visualize
    
    @face_model_points_3d.setter
    def face_model_points_3d(self, value):
        self._face_model_points_3d = value
        
    @left_eye_ball_center.setter
    def left_eye_ball_center(self, value):
        self._left_eye_ball_center = value
    
    @right_eye_ball_center.setter
    def right_eye_ball_center(self, value):
        self._right_eye_ball_center = value
    
    @facial_landmark_index.setter
    def facial_landmark_index(self, value):
        self._facial_landmark_index = value
    
    @camera_matrix.setter
    def camera_matrix(self, value):
        self._camera_matrix = value
    
    @device.setter
    def device(self, value):
        self._device = value
    
    @visualize.setter
    def visualize(self, value):
        self._visualize = value
        
    def find_device(self, max_try: int = 10):
        for device in range(max_try):
            cap = cv2.VideoCapture(device)
            while cap.isOpened():
                success, _ = cap.read()
                if success:
                    return device
        return -1

    def find_camera_matrix(self, frame):
        focal_length = frame.shape[1]
        center = (frame.shape[1] / 2, frame.shape[0] / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
            dtype="double",
        )

        return camera_matrix

    def find_face_num(self, max_try=100):
        mp_face_detection = mp.solutions.face_detection
        with mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5) as face_detection:
            cap = cv2.VideoCapture(self._device)
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

    def set_camera_matrix(self, camera_matrix):
        self._camera_matrix = camera_matrix

    def run(self, max_num_faces=1):
        mp_face_mesh = mp.solutions.face_mesh  # initialize the face mesh model
        assert self._device >= 0
        cap = cv2.VideoCapture(self._device)  # chose camera index (try 1, 2, 3)
        with mp_face_mesh.FaceMesh(
            max_num_faces=max_num_faces,  # number of faces to track in each frame
            refine_landmarks=True,  # includes iris landmarks in the face mesh model
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as face_mesh:
            while cap.isOpened():
                success, frame = cap.read()
                if self._camera_matrix is None:
                    self._camera_matrix = self.find_camera_matrix(frame)
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
                        except TypeError as error:
                            print(f'TypeError: {error}')
                            continue
                        if self._visualize:
                            self.draw(frame, left_pupil, gaze_left_eye)
                            self.draw(frame, right_pupil, gaze_right_eye)
                cv2.imshow("output window", frame)
                if cv2.waitKey(2) & 0xFF == 27:
                    break
        cap.release()

    def calculate_head_eye_poses(
        self, frame: np.ndarray, points: np.ndarray, gaze_distance=10
    ):

        frame_height, frame_width, _ = frame.shape
        # Mediapipe points are normalized to [-1, 1].
        # Image points holds the landmark points in terms of
        # image coordinates
        image_points = np.array(
            [
                (
                    points.landmark[ind].x * frame_width,
                    points.landmark[ind].y * frame_height,
                )
                for ind in self._facial_landmark_index
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
            self._face_model_points_3d,
            image_points,
            self._camera_matrix,
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
            image_points_ext, self._face_model_points_3d
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
                self._left_eye_ball_center
                + (pupil_world_cord[0] - self._left_eye_ball_center) * gaze_distance,
                self._right_eye_ball_center
                + (pupil_world_cord[1] - self._right_eye_ball_center) * gaze_distance,
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
                self._camera_matrix,
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
                self._camera_matrix,
                dist_coeffs,
            )
            # project 3D head pose into the image plane
            head_pose, _ = cv2.projectPoints(
                (int(pupil_world_cord[0][0]), int(pupil_world_cord[0][1]), int(40)),
                rotation_vector,
                translation_vector,
                self._camera_matrix,
                dist_coeffs,
            )

            head_pose1, _ = cv2.projectPoints(
                (int(pupil_world_cord[1][0]), int(pupil_world_cord[1][1]), int(40)),
                rotation_vector,
                translation_vector,
                self._camera_matrix,
                dist_coeffs,
            )
            # correct gaze for head rotation
            gaze_left_eye = (
                left_pupil
                + (gaze_direction_left_eye[0][0] - left_pupil)
                - (head_pose[0][0] - left_pupil)
            )

            gaze_right_eye = (
                right_pupil
                + (gaze_direction_right_eye[0][0] - right_pupil)
                - (head_pose1[0][0] - right_pupil)
            )

            return (left_pupil, right_pupil), (gaze_left_eye, gaze_right_eye)

    def draw(self, frame, left_pupil, gaze):
        # Draw gaze line into screen
        p1 = (int(left_pupil[0]), int(left_pupil[1]))
        p2 = (int(gaze[0]), int(gaze[1]))
        cv2.line(frame, p1, p2, (0, 0, 255), 2)

        v1, v2 = self.calculate_arrowhead(p1, p2)
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
# g.set_camera_matrix(
#     np.array(
#         [
#             [1394.6027293299926, 0, 995.588675691456],
#             [0, 1394.6027293299926, 599.3212928484164],
#             [0, 0, 1],
#         ]
#     )
# )
# # print(g.get_face_num())
# g.run()