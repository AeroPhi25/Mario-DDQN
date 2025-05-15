# Mario-DDQN
Contains sample code for training an AI that plays Mario.

# Description of mobile_mario.ipynb and non_mobile_mario.ipynb:

Trains an AI to play Mario using DDQN and linear decay for epsilon.
The only difference between these codes is the actions the agent can perform.
The mobile agent can move in all directions, while the non-mobile agent can only move right and jump right.

# Instructions for mobile_mario.ipynb and non_mobile_mario.ipynb:
1. Make sure the environment used is utilizing python 3.11.9
2. Install Jupyter Notebook
3. Install PyTorch with CUDA support
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
4. Install remaining dependencies

     pip install gym==0.26.2

     pip install gym-super-mario-bros==7.4.0

     pip install nes-py==8.2.1

     pip install matplotlib tqdm tensordict

     pip install torchrl

     pip install numpy==1.23.5

5. Run the code (it should now run without issue)

For further customization:

Mobility can be adjusted by editing the line:

     env = JoypadSpace(env, [["right"], ["right", "A"], ["right", "B"], ["right", "A", "B"], ["left"], ["left", "A"], ["left", "A", "B"], ["A"], ["B"]])

The episode quantity can be edited in the last cell:

     episodes = _____

For several different runs, change the save directory of the results in the last cell to avoid overwriting:

     save_dir = Path("____NAME_____") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")



# Description of Mario DDQN_NDDQN.py:

This project trains two reinforcement-learning agents to play Super Mario Bros:
- Dueling DQN (DDQN): baseline agent that relies on an ε-greedy policy with linear ε-decay.  
- Noisy Dueling DQN (NDDQN):  adds learnable “noisy” layers that replace ε-greedy exploration.
Each agent is tested with two action sets.

The goal is to see which exploration trick scores hiagher, learns faster and wastes fewer frames.


# Instructions for Mario DDQN_NDDQN.py:

1. Create a python 3.11 environment (if needed)
        python -m venv .venv
          # Windows: .venv\Scripts\activate
          # macOS/Linux: source .venv/bin/activate
2. Install Jypyter & Core Libraries
        pip install notebook
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118            #GPU build #(CPU-only?  Omit the --index-url flag.)
3. Add project dependencies
        pip install gym==0.26.2
        pip install gym-super-mario-bros==7.4.0
        pip install numpy matplotlib pygame tqdm
   
4. Run the training #number of episodes can be adjusted for better performance.
5. Check results
         #Checkpoints and replay buffers → ./checkpoints/
         #Training curves and gameplay videos → ./outputs/      
