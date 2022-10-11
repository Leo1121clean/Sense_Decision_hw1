from PIL import Image
import numpy as np
import habitat_sim
from habitat_sim.utils.common import d3_40_colors_rgb
import cv2
import open3d as o3d
import math


test_scene = "apartment_0/habitat/mesh_semantic.ply"
sim_settings = {
    "scene": test_scene,  # Scene path
    "default_agent": 0,  # Index of the default agent
    "sensor_height": 1,  # Height of sensors in meters, relative to the agent
    "width": 512,  # Spatial resolution of the observations
    "height": 512,
    "sensor_pitch": 0,  # sensor pitch (x rotation in rads)
}

#將rgb轉換成bgr
def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

#將深度圖資訊轉換成圖像
def transform_depth(image):
    depth_img = (image / 10 * 255).astype(np.uint8)
    return depth_img

#語義圖像(本次作業未使用)
def transform_semantic(semantic_obs):
    semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
    semantic_img.putpalette(d3_40_colors_rgb.flatten())
    semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
    semantic_img = semantic_img.convert("RGB")
    semantic_img = cv2.cvtColor(np.asarray(semantic_img), cv2.COLOR_RGB2BGR)
    return semantic_img

#建立sensor的函式
def make_simple_cfg(settings):
    # simulator backend
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene"]
    # agent
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    
    
    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
    rgb_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    rgb_sensor_spec.orientation = [
        settings["sensor_pitch"],
        0.0,
        0.0,
    ]
    rgb_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    
    # a BEV sensor, to the agent
    bev_sensor_spec = habitat_sim.CameraSensorSpec()
    bev_sensor_spec.uuid = "BEV_sensor"
    bev_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    bev_sensor_spec.resolution = [settings["height"], settings["width"]]
    bev_sensor_spec.position = [0.0, settings["sensor_height"], -1.5]
    bev_sensor_spec.orientation = [
        -np.pi/2,
        0.0,
        0.0,
    ]
    bev_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    #depth snesor
    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    depth_sensor_spec.orientation = [
        settings["sensor_pitch"],
        0.0,
        0.0,
    ]
    depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    #semantic snesor
    semantic_sensor_spec = habitat_sim.CameraSensorSpec()
    semantic_sensor_spec.uuid = "semantic_sensor"
    semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
    semantic_sensor_spec.position = [1.0, settings["sensor_height"], 0.0]
    semantic_sensor_spec.orientation = [
        settings["sensor_pitch"],
        0.0,
        0.0,
    ]
    semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    agent_cfg.sensor_specifications = [rgb_sensor_spec,bev_sensor_spec, depth_sensor_spec, semantic_sensor_spec]

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])
    
#此函式包括sensor控制、圖像輸出及位姿資訊
def navigateAndSee(action=""):
    if action in action_names :
        observations = sim.step(action)
        #print("action: ", action)

        cv2.imshow("RGB", transform_rgb_bgr(observations["color_sensor"]))
        cv2.imshow("BEV", transform_rgb_bgr(observations["BEV_sensor"]))
        #cv2.imshow("depth", transform_depth(observations["depth_sensor"]))
        #cv2.imshow("semantic", transform_semantic(observations["semantic_sensor"]))
        agent_state = agent.get_state()
        sensor_state = agent_state.sensor_states['color_sensor']
        print("camera pose: x y z rw rx ry rz")
        print(sensor_state.position[0],sensor_state.position[1],
              sensor_state.position[2],  sensor_state.rotation.w,
              sensor_state.rotation.x, sensor_state.rotation.y, sensor_state.rotation.z)

        return transform_rgb_bgr(observations["color_sensor"]),transform_rgb_bgr(observations["BEV_sensor"]), transform_depth(observations["depth_sensor"]), sensor_state.position[0],sensor_state.position[1],sensor_state.position[2]


count = 0 #紀錄圖片張數(data size)
save_count = 0 #儲存資料的計數器(每幾步存一次data)

if __name__=="__main__":

    ##################啟動open3d##################
    cfg = make_simple_cfg(sim_settings)
    sim = habitat_sim.Simulator(cfg)


    # initialize an agent
    agent = sim.initialize_agent(sim_settings["default_agent"])

    # Set agent state
    agent_state = habitat_sim.AgentState()
    agent_state.position = np.array([0.0, 0.0, 0.0])  # agent in world space
    agent.set_state(agent_state)

    # obtain the default, discrete actions that an agent can perform
    # default action space contains 3 actions: move_forward, turn_left, and turn_right
    action_names = list(cfg.agents[sim_settings["default_agent"]].action_space.keys())
    print("Discrete action space: ", action_names)


    FORWARD_KEY="w"
    LEFT_KEY="a"
    RIGHT_KEY="d"
    FINISH="f"

    SAVE_FRONT="q"
    SAVE_BEV="e"

    print("#############################")
    print("use keyboard to control the agent")
    print(" w for go forward  ")
    print(" a for turn left  ")
    print(" d for trun right  ")
    print(" f for finish and quit the program")
    print("#############################")
    ##################啟動open3d##################
    
    action = "move_forward"
    s = navigateAndSee(action) #將初始位置資料存至s
    
    with open('ground_xyz.txt', 'w') as outfile:
        outfile.truncate()
    
    ##################使用鍵盤按鍵持續在模擬環境中移動及存圖##################
    while True:
        
        #每行走2步，會儲存一次rgb及depth圖像，同時記錄當下座標及更新圖像張數
        if save_count%2 == 0:
        
            count = count + 1
            save_count = 0
            
            #儲存rgb及depth圖像
            cv2.imwrite('img/rgb/rgb' + str(count) + '.png',s[0])
            cv2.imwrite('img/depth/depth' + str(count) + '.png',s[2])
            print("Save image " + str(count))
            
            #記錄當下座標記錄當下座標
            with open('ground_xyz.txt', 'a') as outfile:
                outfile.write(str(s[3]) + ' ' + str(s[4]) + ' ' + str(s[5]) + ' \n')
            
            #更新圖像張數
            with open('count.txt', 'w') as outfile:
                outfile.write(str(count))

        keystroke = cv2.waitKey(0)
        if keystroke == ord(FORWARD_KEY):
            action = "move_forward"
            s = navigateAndSee(action)
            print("action: FORWARD")
            save_count = save_count + 1
        elif keystroke == ord(LEFT_KEY):
            action = "turn_left"
            s = navigateAndSee(action)
            print("action: LEFT")
            save_count = save_count + 1
        elif keystroke == ord(RIGHT_KEY):
            action = "turn_right"
            s = navigateAndSee(action)
            print("action: RIGHT")
            save_count = save_count + 1
        elif keystroke == ord(FINISH):
            print("action: FINISH")
            break
        elif keystroke == ord(SAVE_FRONT):  #get Front_view image
            cv2.imwrite('Front_view.png',s[0])
            print("Save front image.")
        elif keystroke == ord(SAVE_BEV):    #get BEV_view image
            cv2.imwrite('BEV_view.png',s[1])
            print("Save BEV image.")
        else:
            print("INVALID KEY")
            continue
    ##################使用鍵盤按鍵持續在模擬環境中移動及存圖##################