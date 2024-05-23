import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from mediapipe import solutions
import uc

# USER PREFERENCES
MAIN_HAND = "Right"
GLOVES = True
IN_CONSOLE = True

# * Robot restrictions
VMIN = 10
VMAX = 170
VMAX_CLAW = 255
VMIN_CLAW = 100
L1 = 6
L2 = 5
L3 = 6.5
REACH = L2 + L3
voltages = [VMAX-VMIN, VMAX-VMIN, VMAX-VMIN, VMAX_CLAW-VMIN_CLAW]
locked = False
locked_timer = 0


# * Constants for rendering the hand landmarks
MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green

# * Hand Positioning Constants
SCREEN_MARGINS = ((0.3, 0.95), (0.1, 0.9))
if MAIN_HAND == "Left":
    SCREEN_MARGINS = ((1 - SCREEN_MARGINS[0][1], 1 - SCREEN_MARGINS[0][0]), SCREEN_MARGINS[1])


if IN_CONSOLE:
    def transform_camera(image):
        image = cv2.rotate(image, cv2.ROTATE_180)
        return image
else:
    def transform_camera(image):
        image = cv2.flip(image, 1)
        return image



def draw_landmarks_on_image(rgb_image, detection_result, lookup_tables, ax = None, main_hand = MAIN_HAND):
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

        second_hand_landmarks = hand_landmarks_list[1 - idx]

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

        # Define the origin of the graph (the intersection of the axes)
        origin = (int(width * (SCREEN_MARGINS[0][0] + (SCREEN_MARGINS[0][1] - SCREEN_MARGINS[0][0]) / 2)), int(height * SCREEN_MARGINS[1][1]))

        # Draw the x-axis
        cv2.line(annotated_image, (0, origin[1]), (width, origin[1]), (0, 255, 0), 2)

        # Draw the y-axis
        cv2.line(annotated_image, (origin[0], 0), (origin[0], height), (0, 255, 0), 2)

        # display locked status
        cv2.putText(annotated_image, "Locked" if locked else "Unlocked", (int(width*0.1), int(height*0.1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if not locked else (0, 0, 255), 2)

        move_robot(lookup_tables, np.array([z_coordinates, x_coordinates, y_coordinates]), hand_mid_points[1 - idx][1], second_hand_landmarks)

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

def deg2rad(deg):
    return deg*np.pi/180

def gen_lt():
    t_alfa = np.zeros((VMAX - VMIN + 1, VMAX - VMIN + 1))
    t_z = np.zeros((VMAX - VMIN + 1, VMAX - VMIN + 1))
    for n in range(VMIN, VMAX + 1):
        for m in range(VMIN, VMAX + 1):
            #n is V1, m is V2
            nm = [n*-0.0211 + 3.5848, m*0.0178 + 0.3473]
            t_alfa[n - VMIN, m - VMIN] = alfa(*nm)
            t_z[n - VMIN, m - VMIN] = z(*nm)


    return [t_alfa, t_z]

def recognize_gesture(points):
    """
    Recognizes the gesture of the hand

    If the locked was changed in the last 10 frames, it will return the current locked status.
    Else, it will change the locked status if the fingers are closed.
    """

    global locked
    global locked_timer

    if locked_timer > 0:
        locked_timer -= 1
        return locked
    points = [(points.landmark[i].x, points.landmark[i].y, points.landmark[i].z) for i in range(len(points.landmark))]

    # Set up hysterisis thresholds
    threshold_to_activate = 0.7

    if measure_fingers(np.array(points).T) >= threshold_to_activate:
        return locked
    locked_timer = 10
    return not locked

def move_robot(lookup_tables, main_points, second_hand_bottom, second_hand_points):
    global voltages, locked 

    fingers = measure_fingers(main_points)
    hand_mid_point = (main_points[:, 5] + main_points[:, 9] + main_points[:, 13] + main_points[:, 17]+4*main_points[:, 0]) / 8


    z = (second_hand_bottom + 1 - SCREEN_MARGINS[1][0])/(SCREEN_MARGINS[1][1] - SCREEN_MARGINS[1][0])

    x = ((hand_mid_point[1] - SCREEN_MARGINS[0][0])/(SCREEN_MARGINS[0][1] - SCREEN_MARGINS[0][0]) - 0.5) * 2
    y = (hand_mid_point[2] - SCREEN_MARGINS[1][0])/(SCREEN_MARGINS[1][1] - SCREEN_MARGINS[1][0])

    desired_coords = np.array([x, y, z])

    locked = recognize_gesture(second_hand_points)

    #check if desired_coords is reachable
    if np.any(np.abs(desired_coords) >1) or np.any(np.abs(desired_coords) < 0):
        print("Desired coordinates out of reach", desired_coords)
        return

    desired_coords = desired_coords*REACH

    teta0 = (np.arctan(desired_coords[1]/desired_coords[0])) % np.pi - np.pi/2

    alfa = (desired_coords[1]**2 + desired_coords[0]**2)**0.5

    dist_vector = np.linspace(VMIN,VMAX, VMAX - VMIN+1)
    #make it a table of the same size as the lookup tables
    dist_table_hor = np.tile(dist_vector, (VMAX - VMIN + 1, 1))   - voltages[1]
    dist_table_ver = np.tile(dist_vector, (VMAX - VMIN + 1, 1)).T - voltages[2]

    table_math = (lookup_tables[0]-alfa)**2 + (lookup_tables[1]-desired_coords[2])**2 + 0.001*(dist_table_hor**2 + dist_table_ver**2)
    
    idx = np.argmin(table_math)

    #normalize lookup table
    #table_math = (table_math - np.min(table_math))/(np.max(table_math) - np.min(table_math))
    #table_math = table_math.astype(np.float32)  # convert to CV_32F
    #table_math = cv2.cvtColor(table_math, cv2.COLOR_GRAY2BGR)
    # display the lookup table
    #table_math[idx//(VMAX - VMIN + 1), idx%(VMAX - VMIN + 1)] = [255, 0, 0]
    # cv2.imshow("Lookup Table", cv2.resize(table_math, (400, 400)))    
    # cv2.waitKey(1)


    V0 = int((teta0+1.3158)/0.0147)
    V1 = int(idx // (VMAX - VMIN + 1) + VMIN)
    V2 = int(idx % (VMAX - VMIN + 1) + VMIN)
    if not(locked):
        V3 = int((1 - min(max(fingers - 0.3, 0), 1))*(VMAX_CLAW - VMIN_CLAW) + VMIN_CLAW)
    else:
        V3 = voltages[3]

    voltages = [V0, V1, V2, V3]
    uc.send(voltages)


def main():
    # * Initialize Mediapipe Hands model
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

    # * Open webcam
    cap = cv2.VideoCapture(0)

    lookup_tables = gen_lt()

    while cap.isOpened():

        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # * Flip the image horizontally for a later selfie-view display
        image = transform_camera(image)

        # * Convert the BGR image to RGB
        if not GLOVES:
            image_to_process = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_to_process = image

        # * Detect hand landmarks from the input image
        results = hands.process(image_to_process)

        if results.multi_hand_landmarks:
            # * Visualize the hand landmarks
            annotated_image = draw_landmarks_on_image(image, results, lookup_tables)
            cv2.imshow('Hand Landmarks Detection', annotated_image)
            plt.pause(0.0001)  # Necessary for real-time updating of the plot

        else:
            cv2.imshow('Hand Landmarks Detection', image)

        if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit
            break

    # * Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    """
    
    pr = cProfile.Profile()
    pr.enable()


    try:
        stats = cProfile.run("main()")
    except Exception:
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
        ps.print_stats()

        with open('profile.txt', 'w+') as f:
            f.write(s.getvalue())
    """
    main()