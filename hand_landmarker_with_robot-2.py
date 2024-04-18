import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from mediapipe import solutions
import claw_sim
import time
import uc

# * Robot restrictions
VMIN = 10
VMAX = 170
VMAX_CLAW = 255
VMIN_CLAW = 100
L1 = 6
L2 = 4.5
L3 = 6.5
REACH = L2 + L3
MAIN_HAND = "Right"
GLOVES = False

# * Constants for rendering the hand landmarks
MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green


def draw_landmarks_on_image(rgb_image, detection_result, robot, lookup_tables, ax = None, main_hand = MAIN_HAND):
    hand_landmarks_list = detection_result.multi_hand_landmarks
    handedness_list = detection_result.multi_handedness
    annotated_image = np.copy(rgb_image)

    if hand_landmarks_list and handedness_list and len(handedness_list) > 1:
        hand_mid_points = [(hand_landmarks.landmark[0].x, -hand_landmarks.landmark[0].y) for hand_landmarks in hand_landmarks_list]
        idx = int(hand_mid_points[0][0] < hand_mid_points[1][0])
        if main_hand != "Right":
            idx = 1-idx

        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # * Draw the hand landmarks.
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style())

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks.landmark]
        y_coordinates = [-landmark.y for landmark in hand_landmarks.landmark]
        z_coordinates = [landmark.z for landmark in hand_landmarks.landmark]
        text_x = int(min(x_coordinates) * width)
        text_y = int(max(y_coordinates) * -1 * height) - MARGIN

        # * Draw handedness (left or right hand) on the image.
        cv2.putText(annotated_image, f"{handedness.classification[0].label[0]}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

        move_robot(robot, lookup_tables, np.array([z_coordinates, x_coordinates, y_coordinates]), hand_mid_points[1 - idx][1])

    return annotated_image


def dist(a, b):
    return (np.sum((a - b)**2))**0.5


def finger_len(points, joints):
    total = 0
    for idx in range(len(joints)-1):
        total += dist(points[:, joints[idx]], points[:, joints[idx+1]])
    return total


def measure_fingers(points):
    palm_size = np.array([dist(points[:, 5], points[:, 17]), dist(points[:, 0], points[:, 17])])

    return dist(points[:, 8], points[:, 4])/palm_size[0]


def alfa(theta1, theta2):
    return L1*np.cos(theta1) + L2*np.cos(theta1 + theta2 - np.pi)


def z(theta1, theta2):
    return L1*np.sin(theta1) + L2*np.sin(theta1 + theta2 - np.pi)


def gen_lt():
    t_alfa = np.zeros((VMAX - VMIN + 1, VMAX - VMIN + 1))
    t_z = np.zeros((VMAX - VMIN + 1, VMAX - VMIN + 1))

    for n in range(VMIN, VMAX + 1):
        for m in range(VMIN, VMAX + 1):
            t_alfa[n - VMIN, m - VMIN] = alfa(n/255*np.pi, m/255*np.pi)
            t_z[n - VMIN, m - VMIN] = z(n/255*np.pi, m/255*np.pi)

    return [t_alfa, t_z]


def move_robot(robot, lookup_tables, main_points, second_hand_bottom):

    fingers = measure_fingers(main_points)
    hand_mid_point = (main_points[:, 5] + main_points[:, 9] + main_points[:, 13] + main_points[:, 17]+4*main_points[:, 0]) / 8

    min_height = 0.1
    max_height = 10
    screen_margins = (0.1, 0.8)

    z = max(0, min((second_hand_bottom + 1 - screen_margins[0])/(screen_margins[1] - screen_margins[0]), 1)) * (max_height-min_height) + min_height

    desired_coords = np.array([2*(hand_mid_point[1]-0.5)*REACH, (hand_mid_point[2]+1) * REACH,z])

    teta0 = (np.arctan(desired_coords[1]/desired_coords[0])/np.pi) % 1

    alfa = (desired_coords[1]**2 + desired_coords[0]**2)**0.5

    table_math = (lookup_tables[0]-alfa)**2 + (lookup_tables[1]-desired_coords[2])**2

    idx = np.argmin(table_math)

    V0 = int((teta0)*(VMAX - VMIN) + VMIN)
    V1 = int(idx // (VMAX - VMIN + 1) + VMIN)
    V2 = int(idx % (VMAX - VMIN + 1) + VMIN)
    V3 = int((1 - np.arcsin(min(max(fingers - 0.3, 0), 1))*2/np.pi)*(VMAX_CLAW - VMIN_CLAW) + VMIN_CLAW)

    # TODO: Comment this line and send Vs to Arduino
    claw_sim.update_arm_plot(robot, random_v=False, voltages=[V0, V1, V2, V3])
    print([V0, V1, V2, V3])
    uc.send([V0, V1, V2, V3])
    

def main():
    # * Initialize Mediapipe Hands model
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

    # * Open webcam
    cap = cv2.VideoCapture(0)

    robot = claw_sim.Sunfounder()
    lookup_tables = gen_lt()

    while cap.isOpened():

        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # * Flip the image horizontally for a later selfie-view display
        image = cv2.flip(image, 1)

        # * Convert the BGR image to RGB
        if not GLOVES:
            image_to_process = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_to_process = image

        # * Detect hand landmarks from the input image
        results = hands.process(image_to_process)

        if results.multi_hand_landmarks:
            # * Visualize the hand landmarks
            annotated_image = draw_landmarks_on_image(image, results, robot, lookup_tables)
            cv2.imshow('Hand Landmarks Detection', annotated_image)
            plt.pause(0.001)  # Necessary for real-time updating of the plot

        else:
            cv2.imshow('Hand Landmarks Detection', image)

        if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit
            break

    # * Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
