import numpy as np
from gaze import Gazetimation

gz = Gazetimation()

def test_face_model_points_3d():
    assert np.all(gz.face_model_points_3d == np.array(
                [
                    (0.0, 0.0, 0.0),
                    (0, -63.6, -12.5),
                    (-43.3, 32.7, -26),
                    (43.3, 32.7, -26),
                    (-28.9, -28.9, -24.1),
                    (28.9, -28.9, -24.1),
                ]
            ))
    gz.face_model_points_3d = True
    assert gz.face_model_points_3d == True

def test_left_eye_ball_center():
    assert np.all(gz.left_eye_ball_center == np.array([[29.05], [32.7], [-39.5]]))
    gz.left_eye_ball_center = True
    assert gz.left_eye_ball_center == True

def test_right_eye_ball_center():
    assert np.all(gz.right_eye_ball_center == np.array([[-29.05], [32.7], [-39.5]]))
    gz.right_eye_ball_center = True
    assert gz.right_eye_ball_center == True

def test_facial_landmark_index():
    assert gz.facial_landmark_index == [4, 152, 263, 33, 287, 57]
    gz.facial_landmark_index = True
    assert gz.facial_landmark_index == True

def test_camera_matrix():
    assert gz.camera_matrix == None
    gz.camera_matrix = True
    assert gz.camera_matrix == True
    
def test_device():
    assert gz.device == 0
    gz.device = 10
    assert gz.device == 10
    
def test_visualize():
    assert gz.visualize == True
    gz.visualize = False
    assert gz.visualize == False

def test_find_device():
    assert gz.find_device(max_try = 0) == -1