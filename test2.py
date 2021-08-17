#from model10 import *
#from movement import *
from cellObjectTest import *
from mesa.visualization.modules import CanvasGrid, ChartModule, TextElement
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter




#change to multigrid where oxygen expands to any open space adjacent to it
#but only oxygen can occupy 


#have tail agents that follow a head agent
#do this by haeing an iditieifer to a tail agent and follow the idifiers to get to the end

def convert_rgb_to_hex(int):
   inverse = 255 - int
   return "#%02x%02x%02x" % (inverse, inverse, inverse)


def agent_portrayal(agent):
   portrayal = {"Shape": "circle",
                 "Color": "red",
                 "Filled": "true",
                 "Layer": 0,
                 "r": 0.5}

   if isinstance(agent, LarvaAgent):
      portrayal["Layer"] = 3
      portrayal["Shape"] = "rect"
      portrayal["w"] = 1
      portrayal["h"] = 1
      portrayal["Color"] = "#B7593F"
      
      #"#050A30"  # F8D05A"  # "#C03221"  # "#fdf9a6"
   if isinstance(agent, LarvaTail):
      portrayal["Layer"] = 2
      portrayal["Color"] = "#DC3F2E"#"#e3e095"
   
   if isinstance(agent,Oxygen):
      portrayal["Color"] = "#545E75"#"blue"
      portrayal["Shape"] = "rect"
      portrayal["w"] = 1
      portrayal["h"] = 1

      if agent.traversable == True:
         portrayal["Color"] = "#83C3AF"

      if agent.depleted == True:
         portrayal["Color"] = "#b5c2c4"
         #portrayal["r"] = 0.2
         portrayal["filled"] = "false"
        # "#44B5BA"

      portrayal["Color"] = convert_rgb_to_hex(min(agent.enter_count,255))
      #print(str(portrayal["Color"]))

   if isinstance(agent,Food):
      portrayal["Color"] = "#E3D5BC"  # E9CD9E"
      #"#F2D0A4"  # 'brown'
      portrayal["Layer"] = 1
      portrayal["w"] = 1
      portrayal["h"] = 1
      portrayal["Shape"] = "rect"

      #if agent.check_for_update:
         #portrayal["Color"] = "#Ad7335"

      #if agent.is_poop:
         #portrayal["Color"] = "#Ad7335"

   if isinstance(agent,Poop):

      portrayal["Color"] = "#86857B"  # 65665C"  # Ad7335"

   
   return portrayal


larva_params = {
    "breathe_amount": UserSettableParameter("slider", "Oxy per breath", 1.5, 0, 10, 0.1),
    "eat_amount": UserSettableParameter("slider", "Max food per eat action", 1.5, 0, 10, 0.1),
    "max_oxy": UserSettableParameter("slider", "Max storeable oxy", 1.5, 0, 10, 0.1),
    "max_food": UserSettableParameter("slider", "Max storeable food", 1.5, 0, 10, 0.1),
    "low_oxy": UserSettableParameter("slider", "Low oxy threshold", 3.0, 0, 10, 0.1),
    "low_food": UserSettableParameter("slider", "Low food threshold", 3.0, 0, 10, 0.1),
    "eat_efficiency": UserSettableParameter("slider", "Food eating efficiency", 0.5, 0, 1, 0.1),
    #"conversion_ratio": UserSettableParameter("slider", "Energy conversion ratio (Food:Oxy)", 1.0, 0, 10, 0.05),

}

#larva_params.low_oxy = UserSettableParameter(
#    "slider", "Low oxy threshold", 3.0, 0, larva_params["max_oxy"].value, 0.1)
#larva_params.low_food = UserSettableParameter(
#    "slider", "Low food threshold", 3.0, 0, larva_params["max_food"].value, 0.1)

model_params = {
   "height": 30,
   "width": 60,
   "N": 10,
   #"larva_params" : larva_params,
   "breathe_amount" : larva_params["breathe_amount"],
   "eat_amount": larva_params["eat_amount"],
   "eat_efficiency": larva_params["eat_efficiency"],
   "low_oxy": larva_params["low_oxy"],
   "static_text" : UserSettableParameter(
        'static_text', value="This is a descriptive textbox"),
   #"density": UserSettableParameter("slider", "Agent density", 0.8, 0.1, 1.0, 0.1),
   #"minority_pc": UserSettableParameter("slider", "Fraction minority", 0.2, 0.00, 1.0, 0.05),
   #"homophily": UserSettableParameter("slider", "Homophily", 3, 0, 8, 1)
}




grid = CanvasGrid(agent_portrayal, model_params["width"], 
                  model_params["height"], 500, 500)


server = ModularServer(LarvaModel,
                       [grid],
                       "Test Model",
                       #{"N": model_params["number"], "width": 20, "height": 40},
                       model_params)
server.port = 8521 # The default
server.launch()

#allow them to eat and brethe from anywhere around them
#optimize food needs check boolean    X ish
#spread uniformly
#swap food and oxy importance?? if else   X
#set defualt move to eat?? if they dont need oxy and its possible, eat X
#stomach capacity??? is there a max, do they needd to stop eating
#digestive enzymes, presecne of enzyme makes some food eatable

#go back up randomly???? to top 3 until can breathe again X
#only eat at three at bottom X
#find facing direction
#combind poop and food (quality metric)
# can food close in on empty space?? spreading??
#sensor for bad food???
#digging as an action push stuff to top or sides???
#put digging contents on side not directy on top
#change from isinstance ZX
#dont eat poop??








#if digging push away (diffusion)
# if poop on top, want to go up
# watch in neighborhood
# maybe change poop to 2 fall not 1