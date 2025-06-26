import cv2
import numpy as np
from view_transformer.view_transformer import draw_radar_view

# Dummy frame (black image)
frame = np.zeros((720, 1280, 3), dtype=np.uint8)

# Dummy tracks for one frame
tracks = {
    'players': [
        {
            1: {'position_transformed': [10, 34], 'team': 0},
            2: {'position_transformed': [50, 60], 'team': 1},
            3: {'position_transformed': [80, 20], 'team': 0},
            4: {'position_transformed': [30, 50], 'team': 1},
        }
    ],
    'ball': [
        {
            1: {'position_transformed': [52, 34]}
        }
    ]
}

# Draw radar view on the dummy frame
output = draw_radar_view(frame.copy(), tracks, frame_num=0)

# Save and show the result
cv2.imwrite("test_radar_output.png", output)
# cv2.imshow("Radar Test", output)
# cv2.waitKey(0)
# cv2.destroyAllWindows()