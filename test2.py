from model8 import *
from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer




#change to multigrid where oxygen expands to any open space adjacent to it
#but only oxygen can occupy 


#have tail agents that follow a head agent
#do this by haeing an iditieifer to a tail agent and follow the idifiers to get to the end



def agent_portrayal(agent):
   portrayal = {"Shape": "circle",
                 "Color": "red",
                 "Filled": "true",
                 "Layer": 0,
                 "r": 0.5}

   if isinstance(agent, LarvaAgent):
      portrayal["Layer"] = 1
      portrayal["Color"] = "#C03221"  # "#fdf9a6"
   if isinstance(agent, LarvaTail):
      portrayal["Layer"] = 1
      portrayal["Color"] = "#DC3F2E"#"#e3e095"
   
   if isinstance(agent,Oxygen):
      portrayal["Color"] = "#545E75"#"blue"

      if agent.depleted == True:
         portrayal["Color"] = "#b5c2c4"
      
      if agent.traversable == True:
         portrayal["Color"] = "#83C3AF"  # "#44B5BA"


   if isinstance(agent,Food):
      portrayal["Color"] = "#F2D0A4"  # 'brown'
   
   return portrayal


grid = CanvasGrid(agent_portrayal, 20, 40, 500, 500)


server = ModularServer(LarvaModel,
                       [grid],
                       "Test Model",
                       {"N":2, "width":20, "height":40})
server.port = 8521 # The default
server.launch()


#make them only able to eat what is directly below them
