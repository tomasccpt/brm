import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from mediapipe import solutions
import claw_sim

#robot restrictions
vmin = 14
vmax = 240
vmax_claw = 150




MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green


#make color order the same as the ovelay
colors = []
for k in solutions.drawing_styles.get_default_hand_landmarks_style().keys():
    temp = solutions.drawing_styles.get_default_hand_landmarks_style()[k].color
    temp = [temp[2],temp[1],temp[0]]
    colors.append(temp)

colors = [*colors[:2],*colors[6:9],colors[2],*colors[9:12],colors[3],*colors[12:15],colors[4],*colors[15:18],colors[5],*colors[18:]]




def draw_landmarks_on_image(rgb_image, detection_result, ax = None, main_hand = "Right"):
    hand_landmarks_list = detection_result.multi_hand_landmarks
    handedness_list = detection_result.multi_handedness
    annotated_image = np.copy(rgb_image)


    if hand_landmarks_list and handedness_list and len(handedness_list) > 1:

        hand_mid_points = [ (hand_landmarks.landmark[0].x, -hand_landmarks.landmark[0].y) for hand_landmarks in hand_landmarks_list]
        idx = int(hand_mid_points[0][0] < hand_mid_points[1][0])
        if main_hand != "Right":
            idx = 1-idx


        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]


        # Draw the hand landmarks.
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


        # Draw handedness (left or right hand) on the image.
        cv2.putText(annotated_image, f"{handedness.classification[0].label[0]}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

        
        move_robot(np.array([z_coordinates, x_coordinates, y_coordinates]), hand_mid_points[1 - idx][1])


        # Plot hand landmarks in 3D
        #ax.scatter(z_coordinates ,x_coordinates, y_coordinates, c=np.array(colors)/255, marker='o')
        #ax.set_xlim([-0.5, 0.5])
        #ax.set_ylim([0, 1])
        #ax.set_zlim3d([-1, 0])
                
        #define viewing angle
        #ax.view_init(elev=0, azim=0)

    return annotated_image

def dist(a, b):
    return (np.sum((a - b)**2))**0.5

def finger_len(points, joints):
    total = 0
    for idx in range(len(joints)-1):
        total += dist(points[:,joints[idx]],points[:,joints[idx+1]])
    return total




def measure_fingers(points):
    finger_lens = np.array([finger_len(points,[2,3,4]), finger_len(points,[5,6,7,8]), finger_len(points,[9,10,11,12]), finger_len(points,[13,14,15,16]), finger_len(points,[17,18,19,20])])
    palm_size = np.array([dist(points[:,5],points[:,17]),dist(points[:,0],points[:,17])])

    return dist(points[:,8],points[:,4])/palm_size[0]


def gen_lt(l1, l2):
    l1 = 5
    l2 = 5


    def alfa(theta1,theta2):
        return l1*np.cos(theta1) + l2*np.cos(theta1 + theta2 - np.pi)
        
    def z(theta1,theta2):
        return l1*np.sin(theta1) + l2*np.sin(theta1 + theta2 - np.pi)

    t_alfa = np.zeros((vmax - vmin +1, vmax - vmin + 1))
    t_z = np.zeros((vmax - vmin +1, vmax - vmin + 1))

    for n in range(vmin, vmax +1):
        for m in range(vmin, vmax +1):
            t_alfa[n - vmin, m - vmin] = alfa(n/255*np.pi,m/255*np.pi)
            t_z[n - vmin, m - vmin] = z(n/255*np.pi,m/255*np.pi)

    return [t_alfa, t_z]



def move_robot(main_points, second_hand_bottom):
    reach = 5
    
    fingers = measure_fingers(main_points)
    hand_mid_point = (main_points[:,5]+main_points[:,9]+main_points[:,13]+main_points[:,17]+4*main_points[:,0])/8


    min_height = 0.1
    max_height = 10
    screen_margins = (0.1,0.8)

    z = max(0, min((second_hand_bottom + 1 - screen_margins[0])/(screen_margins[1]-screen_margins[0]), 1)) * (max_height-min_height) + min_height



    desired_coords = np.array([2*(hand_mid_point[1]-0.5)*reach, (hand_mid_point[2]+1)*reach,z])


    teta0 = (np.arctan(desired_coords[1]/desired_coords[0])/np.pi)%1

    alfa = (desired_coords[1]**2 + desired_coords[0]**2)**0.5
    
    table_math = (lookup_tables[0]-alfa)**2 + (lookup_tables[1]-desired_coords[2])**2

    idx = np.argmin(table_math)


    V0 = int((teta0)*(vmax_claw - vmin) + vmin)
    V1 = idx//(vmax - vmin +1) + vmin
    V2 = idx%(vmax - vmin +1) + vmin
    V3 = (np.arcsin(min(max(fingers-0.3,0),1))*2/np.pi)*(vmax - vmin) + vmin


    claw_sim.update_arm_plot(robot_arm, random_v=False, voltages=[V0, V1, V2, V3])




# Initialize Mediapipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Open webcam
cap = cv2.VideoCapture(0)

# Create a figure for 3D scatterplot
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')

robot_arm = claw_sim.Sunfounder()
lookup_tables = gen_lt(5,3)

while cap.isOpened():

    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip the image horizontally for a later selfie-view display
    image = cv2.flip(image, 1)

    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect hand landmarks from the input image
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        # Visualize the hand landmarks
        annotated_image = draw_landmarks_on_image(image, results)
        cv2.imshow('Hand Landmarks Detection', annotated_image)
        plt.pause(0.001)  # Necessary for real-time updating of the plot
        #ax.cla()  # Clear the scatter plot for the next frame

    else:
        cv2.imshow('Hand Landmarks Detection', image)

    if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit
        break


# Release resources
cap.release()
cv2.destroyAllWindows()
