from train import DQN
import torch 
import gymnasium as gym 
import time




def play() :
      env = gym.make("CartPole-v1" , render_mode = 'human')
      state , _= env.reset()
      device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else 'cpu'
      agent = DQN(obs=len(state) ,actions=env.action_space.n)
      agent.load_state_dict(torch.load("dpn_cartpole.pth" , weights_only=True))
      agent.eval()
      total_reward = 0
      running = True 

      with torch.no_grad() :
        state = torch.tensor(state , dtype=torch.float32 , device=device).unsqueeze(0)
        while running :
            action  = agent(state).max(1).indices.view(1,1)
            observation , reward , terminated , truncated , info = env.step(action.item())
            total_reward += reward
            if terminated or truncated :
                running = False
            state =  torch.tensor(observation , dtype=torch.float32 , device=device).unsqueeze(0)

      print("total reward :" ,total_reward)    


if __name__ == "__main__" :
    play()  


