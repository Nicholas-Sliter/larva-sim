import math
import random
from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from collections import namedtuple
import typing


Point = namedtuple('Point', 'x y')
PotentialMove = namedtuple("PotentialMove", ["pos", "content"])
Cost = namedtuple('Cost', ['food', 'oxygen'])
Reward = namedtuple('Reward', ['food', 'oxygen'])
Move = namedtuple(
    "Move", ["pos", "action", "location", "cost", "reward", "value"])
RankedMove = namedtuple("Rank", ["move", "rank"])
Params = namedtuple("Parameters", ["breathe_amount", "eat_amount", "eat_efficiency", "max_food", "max_oxy",
   "low_food", "low_oxy"] )


class LarvaParams():
   def __init__(self,dict):
      '''self.EAT_AMOUNT = 0
      self.EAT_EFFICIENCY = 0
      self.BREATHE_AMOUNT = 0
      self.MAX_FOOD = 0 #6
      self.MAX_OXY = 0 #6
      self.LOW_FOOD = 0
      self.LOW_OXY = 0'''
      #self.POOP_THRESHOLD = 5

      self.PARAMETERS = {}

      for param in dict:
         #print(param)
         setattr(self, param, dict[param])




class NeighborHoodObject():
   def __init__(self,model:Model,pos:tuple,center=False,moore=False):
      self.model = model
      self.pos = pos
      self.neighbor_pos = self.neighbors(center,moore)

      #List of moves where actions can take place as prerequiste conditions are met
      self.edible_list = list()
      self.breatheable_list = list()
      self.moveable_list = list()
      self.swapable_list = list()

      # make this a dictionary acessable by pos
      self.neighbors = {}
      for pos in self.neighbor_pos:
         cell = CellObject(self.model,pos)
         self.neighbors[pos] = cell

   def update(self, center=False,moore=False):
      self.get_neighbors(center,moore)
      self.get_edible()
      self.get_breatheable()
      self.get_moveable()

   def get_neighbors(self, center, moore):
      self.neighbor_pos = self.neighbors(center, moore)

      self.neighbors = list()
      for neighbor in self.neighbor_pos:
         cell = CellObject(self.model,neighbor)
         self.neighbors.append(cell)

   def get_cell(self,pos):
      return self.neighbors[pos]

   def get_edible(self):
      self.edible_list = [self.neighbors[neighbor]
                          for neighbor in self.neighbors if self.neighbors[neighbor].is_edible(self.pos)]
      return self.edible_list


   def get_diffuse_targets(self,type="Food"):
      return [self.neighbors[neighbor]
                          for neighbor in self.neighbors if self.neighbors[neighbor].is_diffusible(type)]


   def get_breatheable(self):
      #Need to do this as it iterates through keys not values
      self.breatheable_list = [
          self.neighbors[neighbor] for neighbor in self.neighbors if self.neighbors[neighbor].is_breatheable(self.pos)]

      return self.breatheable_list

   def get_moveable(self,direction="all"):
      #four direction: up, down, all, side

      check_pos = self.pos[1]
      if direction == "up":
         check_pos += 1
      elif direction == "down":
         check_pos -= 1

      if direction=="all":
         self.moveable_list = [self.neighbors[neighbor]
                               for neighbor in self.neighbors if self.neighbors[neighbor].is_moveable()]
      else:
         self.moveable_list = [
             self.neighbors[neighbor] for neighbor in self.neighbors if self.neighbors[neighbor].is_moveable() and neighbor[1] == check_pos]
         #neighbor.pos[1] == check_pos]
      '''elif direction == "up":
         self.moveable_list = [
             neighbor for neighbor in self.neighbors if neighbor.is_moveable() and neighbor.pos[1] == self.pos[1] + 1]
      elif direction == "down":
         self.moveable_list = [
             neighbor for neighbor in self.neighbors if neighbor.is_moveable() and neighbor.pos[1] == self.pos[1] - 1]
      elif direction == "side":
         self.moveable_list = [
             neighbor for neighbor in self.neighbors if neighbor.is_moveable() and neighbor.pos[1] == self.pos[1]]
      '''
      return self.moveable_list


      pass

   def get_swapable(self,direction="all"):

      if direction == "all":
         self.swapable_list = [self.neighbors[neighbor]
                            for neighbor in self.neighbors if self.neighbors[neighbor].Food is not None or self.neighbors[neighbor].Poop is not None]
      else:

         check_pos = self.pos[1]
         if direction == "up":
            check_pos += 1
         elif direction == "down":
            check_pos -= 1

         self.swapable_list = [self.neighbors[neighbor]
                            for neighbor in self.neighbors if  neighbor[1] == check_pos and (self.neighbors[neighbor].Food is not None or self.neighbors[neighbor].Poop is not None)]

      #let poop and food be moved through by swapping positions, let poop be swapped at any side but food can only be swapped from behind

      #just let it swap with whatever it wants

      return self.swapable_list

   def neighbors(self, center=False, moore=True):
      pos = self.pos

      neighbors = self.model.grid.get_neighborhood(
          pos,
          moore=moore,
          include_center=center)

      return neighbors


class CellObject():
   #store these all in a dictionary??
   def __init__(self,model: Model, pos):
      self.model = model
      self.pos = pos
      self.LarvaAgent = None
      self.Food = None
      self.Poop = None
      self.Oxy = None
      self.LarvaTail = None

      contents = self.__cell_contents()
      for agent in contents:
         classname = agent.__class__.__name__
         #classstring = str(classname)

         #setattr(agent,classname)
         setattr(self,classname,agent)



      self.is_solid = False
      if self.LarvaAgent is not None or self.LarvaTail is not None or self.Food is not None or self.Poop is not None:
         self.is_solid = True

   def is_edible(self,agent_pos):
      if self.Food is None:
         return False
      if self.pos[1] == agent_pos[1] - 1:
         if self.LarvaAgent is None and self.LarvaTail is None:
            return True
      return False

   def is_breatheable(self,agent_pos):
      if self.pos[1] == agent_pos[1] + 1:
         if not self.is_solid:
            return True

      return False

   def is_moveable(self):
      if not self.is_solid:
         return True
      return False

   def is_swapable(self,agent_pos):
      #if cell contains food or poop and no larva, let larva swap position to here
      pass


   def is_diffusible(self,type):
      if getattr(self,type) is not None:
         return True
      return False

   def __cell_contents(self):
      return self.model.grid.get_cell_list_contents(self.pos)


class ExtendedAgent(Agent):
   def __init__(self, unique_id: int, model: Model):
      super().__init__(unique_id, model)
      self.kill_flag = False
      self.type = None
      self.enter_count = 0


   def randindex(self,length=1):
      #returns a rand index of a list of given size
      return random.randint(0,length-1)

   def get_neighborhood(self,pos=None,center=False,moore=False):
      if pos is None:
         pos = self.pos
      neighborhood = self.__get_neighborhood_object(pos,center,moore)

      return neighborhood

   def __get_neighborhood_object(self,pos,center,moore):
      #do checks in dictionary using pos tuple as key
      #or maybe only keep cellObj in dictionary
      return NeighborHoodObject(self.model,pos,center,moore)


   def get_position(self):
      return self.pos

   def neighbors(self, pos=None, center=False, moore=True):

      if (pos == None):
         pos = self.pos

      if isinstance(pos, (ExtendedAgent, LarvaAgent, Agent, LarvaTail, Oxygen, Food)):
         pos = pos.pos

      neighbors = self.model.grid.get_neighborhood(
          pos,
          moore=moore,
          include_center=center)

      return neighbors

   def neighbor_contents(self, pos=None, neighbors=None, center=False, moore=True):
      if pos == None and neighbors == None:
         return list()
      if pos == None:
         pos = self.pos
      if neighbors == None:
         neighbors = self.neighbors(pos, center, moore)

      return self.model.grid.get_cell_list_contents(neighbors)

   def cell_contents(self, pos=None):
      if pos == None:
         pos = self.pos
      return self.model.grid.get_cell_list_contents(pos)

   def is_solid(self, pos):
      contents = self.cell_contents(pos)
      bool = False
      for agent in contents:
         if isinstance(agent, Food):
            bool = True
            break
         if isinstance(agent, LarvaAgent):
            bool = True
            break
         if isinstance(agent, LarvaTail):
            bool = True
            break
      return bool

   def is_food(self,pos):
      contents = self.cell_contents(pos)
      for agent in contents:
         if isinstance(agent, Food):
            return True

   def is_poop(self, pos):
      contents = self.cell_contents(pos)
      for agent in contents:
         if isinstance(agent, Poop):
            return True


   def get_food2(self,pos):
      pass


   def get_food(self, pos):
      contents = self.cell_contents(pos)
      for agent in contents:
         if isinstance(agent, Food) and not hasattr(agent,'is_poop'):
            return agent

      return None

   #def get_food(self,cell_obj: CellObject):
   #   return cell_obj.Food

   def get_poop(self, pos):
      contents = self.cell_contents(pos)
      for agent in contents:
         if isinstance(agent, Poop):
            return agent

      return None

   def _remove_agent(self, agent):
      agent.die()

   def die(self):
      #print(str(self), "running self.die")

      if self.kill_flag == False:
         self.kill_flag = True
         self.model.kill_agents.append(self)
         return True

      return False

   def step(self):
      pass


class LarvaAgent(ExtendedAgent):

   def __init__(self,
      unique_id: int,
      model: Model,
      type=None, 
      next=None, 
      pos=None):

      super().__init__(unique_id, model)

      self.LARVA_SIZE = 1
      self.EAT_AMOUNT = self.model.larva_params["eat_amount"]
      self.EAT_EFFICIENCY = self.model.larva_params["eat_efficiency"]
      self.BREATHE_AMOUNT = self.model.larva_params["breathe_amount"]
      self.MAX_FOOD = 6 #6
      self.MAX_OXY = 12 #6
      self.LOW_FOOD = 0
      self.LOW_OXY = 0
      self.POOP_THRESHOLD = 5

      for param in self.model.larva_params:
         setattr(self, param.upper(), self.model.larva_params[param])

      self.type = type
      self.food = 3.0
      self.oxy = 2.0
      self.staged_move = None
      self.eating = None
      self.times_pooped = 0
      self.poop = 0

      self.end = False
      self.previous = None
      self.next = None

      #the agent that the larva is currently watching, if any
      self.watching = None

      #create tail agents here
      previous_agent = self
      for i in range(self.LARVA_SIZE-1):
         endv = False
         if i+2 == self.LARVA_SIZE:
            endv = True
         id = str(self.unique_id) + str(i*5) + str(1000) + str(i+1)
         agent = LarvaTail(id, self.model, self.type, self,
                           previous_agent, next=None, end=endv)
         previous_agent.next = agent
         previous_agent = agent

      #stage moves for all elements in the larva then exucute those moves using a different function

   def stage(self, cell, location):

      if not cell == None:
         #self.staged_move = cell

         if location == 0:
            #go forward
            if (not (self.next == None)):
               move = self.pos
               self.next.stage(move, location)
               #self.staged_move = cell

         else:
            if not self.previous == None:
               #self.staged_move = cell
               move = self.pos
               self.previous.stage(move, location)
               #self.staged_move = cell

         self.staged_move = cell
      """
      Stages changes for tail agents so that they can be executed at once and move with the head without detachment

      """

   def execute(self, direction=0):
      #print("execute running with: ", self.pos,"direction = ",direction)
      #print("staged_move is: " + str(self.staged_move))

      cell = self.staged_move
      #print("executing move from",self.pos,"to",cell)

      #if isinstance(self, LarvaTail) and self.end == False:
      #self.move_to(cell)  # ,suppress_restrictions=True)
      #else:
      self.move_to(cell)  # ,suppress_restrictions=True)

      if direction == 0 and (not self.next == None):
         self.next.execute(direction)
      elif direction == 1 and (not self.previous == None):
         self.previous.execute(direction)

   def sense(self):
      #senses nearby agents and chooses a nearby agent to watch

      #given the set of neighbors, determine nearby agents

      pass

   def __get_food_reward(self, amount, food_agent_amount):
      r = 0

      eat_amount = min(amount, food_agent_amount)
      r = round(min(self.MAX_FOOD, eat_amount), 4)

      return r

   def __get_oxy_reward(self, amount):
      r = 0

      r = round(min(self.MAX_OXY, amount), 4)

      return r

   def add_food(self, amount):
      self.food = min(self.food+amount, self.MAX_FOOD)
      #print("Food is now:", str(self.food))

   def add_oxy(self, amount):
      self.oxy = min(self.oxy+amount, self.MAX_OXY)

   def __validate_cell(self, cell):
      #Validates moves to cells

      contents = self.model.grid.get_cell_list_contents(cell)

      if (contents == None or len(contents) == 0 or self.model.grid.is_cell_empty(cell)):
         return True
      if len(contents) > 1:
         for i in range(len(contents)):
            if self.is_solid(contents[i]):
               return False

   def get_reward(self, content, action):
      #given a move, determine the reward for the move

      reward_food = 0
      reward_oxy = 0

      if action == "move":
         reward_food = 0
         reward_oxy = 0

      elif action == "action:breathe" or action == "action:breathe+move":
         reward_food = 0
         reward_oxy = self.__get_oxy_reward(self.BREATHE_AMOUNT)

      elif action == "action:eat":
         reward_food = self.__get_food_reward(self.EAT_AMOUNT, content.amount)
         reward_oxy = 0

      elif action == "wait":
         reward_food = 0
         reward_oxy = 0

      return Reward(reward_food, reward_oxy)

   def get_possible_steps(self, location=None):
      #first find the pos to use based on location.
      # location == None or location == 0 imply using self.pos
      # location == 1 should use the tail pos

      pos = None

      agent = self.get_agent(location)
      agent_pos = self.get_agent_pos(location)

      #Now get the neighborhood around pos
      neighbors = self.neighbors(agent_pos)
      neighbors_contents = self.neighbor_contents(neighbors=neighbors)

      #define the empty list that will hold all possible moves
      possible_steps = list()

      def create_move(self, content, pos, action, location):

         #find cost, reward, and value
         cost = self.get_cost(action, location)
         reward = self.get_reward(content, action)
         value = self.__get_value(cost, reward)

         #Create the move using the above values and the move namedtuple
         move = Move(pos, action, location, cost, reward, value)

         return move

      def is_actionable(self, pos, content):
         # finds if a given position is actionable by iterating through its contents and returing false if any
         # contents prevent movement.

         def guard_conditions(self, pos, content):
            # if this conditon returns false then the cell contains invalid contents

            def conditions(self, pos, content):
               #the conditions to check for
               if (content == None or self.model.grid.is_cell_empty(pos)):
                  return False
               if isinstance(content, (LarvaAgent, LarvaTail)):
                  return False

               else:

                  return True

            if isinstance(content, list):
               actionable = True
               for element in content:
                  if conditions(self, pos, element) == False:
                     actionable = False

               return actionable
            else:
               return conditions(self, pos, content)

         if guard_conditions(self, pos, content) == False:
            return False
         else:
            return True

      def get_action(self, pos, content, location, agent_pos):

         #create an empty list to append poterntial actions to
         action_list = list()

         if isinstance(content, Oxygen) and content.get_traversable():
            action = "move"
            action_list.append(action)

            # append another for breathing at tail
            if location == 1 and content.depleted == False and pos[1] >= agent_pos[1]:
               action = "action:breathe"
               #If the position differs
               if not pos == agent_pos:
                  action = "action:breathe+move"
               action_list.append(action)

         if isinstance(content, Food) and location == 0 and pos[1] <= agent_pos[1]:
            action = "action:eat"
            action_list.append(action)

         return action_list

      move_list = list()

      for cell, contents in zip(neighbors, neighbors_contents):
         if is_actionable(self, cell, contents):
            #go through each elements in contents and create a move for it

            if isinstance(contents, list):
               for element in contents:

                  action_list = get_action(
                      self, cell, element, location, agent_pos)

                  if (not action_list == None) and len(action_list) >= 1:
                     for action in action_list:
                        move = create_move(
                            self, element, cell, action, location)
                        move_list.append(move)
            else:
               action_list = get_action(
                   self, cell, contents, location, agent_pos)
               if (not action_list == None) and len(action_list) >= 1:
                  for action in action_list:
                     move = create_move(self, contents, cell, action, location)
                     move_list.append(move)

         if not move_list == None:
            possible_steps.extend(move_list)

      return possible_steps

   def __get_value(self, cost, reward):
      def greedy_func(move_cost, move_reward, current_amount, buffer):
         food_cost = move_cost[0]
         food_reward = move_reward[0]

         food_amount = current_amount[0]
         food_buffer = buffer[0]

         food_value = (food_amount + (food_reward - food_cost)) / food_amount

         oxy_cost = move_cost[1]
         oxy_reward = move_reward[1]
         oxy_amount = current_amount[1]
         oxy_buffer = buffer[1]

         oxy_value = (oxy_amount + (oxy_reward - oxy_cost)) / oxy_amount

         value = round((food_value + oxy_value)/2, 4)
         return value

      current_amount = (self.food, self.oxy)
      buffer = (0.2, 0.2)

      return greedy_func(cost, reward, current_amount, buffer)

   def rank(self, head_moves=None, tail_moves=None):
      #ranks the moves according to their value, computed by the value function
      move_list = list()

      if head_moves == None and tail_moves == None:
         return None

      if not tail_moves == None:
         for move in tail_moves:
            move_list.append(move)

      if not head_moves == None:
         for move in head_moves:
            move_list.append(move)

      move_list.sort(reverse=True, key=lambda move: move.value)
      return move_list

   def update_values(self):
      self.BREATHE_AMT = self.model.larva_params.breathe_amount
      self.EAT_AMOUNT = self.model.larva_params.eat_amount

      pass

   def __get_tail_pos(self):
      agent = self
      while not agent.next == None:
         #Returns the tail as agent
         agent = agent.next
      return agent.pos

   def get_cost(self, action, location=None):
      #use location to make moves going up cost more
      OXY_COST = 0
      FOOD_COST = 0

      if location == None:
         location = 0

      if action == "move":
         OXY_COST = 0.3
         FOOD_COST = 0.4
      elif action == "action:eat":
         OXY_COST = 0.3
         FOOD_COST = 0.5
      elif action == "action:breathe":
         OXY_COST = 0.1
         FOOD_COST = 0.2
      elif action == "action:breathe+move":
         OXY_COST = 0.1 + 0.3  # +1.0
         FOOD_COST = 0.2 + 0.4  # +1.0
      elif action == "wait":
         OXY_COST = 0.1
         FOOD_COST = 0.1

      #also do sqrt(2) (apprx) multiplication to moves that are diagonal

      if location == 1 and not action == "action:breathe":
         #to simulate gravity
         OXY_COST *= 1.2
         FOOD_COST *= 1.2

      return Cost(FOOD_COST, OXY_COST)

   def breathe(self):

      agent = self
      if isinstance(self, LarvaTail):
         agent = self.agent

      #agent.oxy = self.add_oxy(self.BREATHE_AMOUNT)
      self.add_oxy(self.BREATHE_AMOUNT)

   def get_movement_cost(self):
      #gets the cost to move from the current position to pos
      OXY_COST = 0.5
      FOOD_COST = 1

      return (FOOD_COST, OXY_COST)

   def apply_cost(self, action):
      agent = self
      if isinstance(self, LarvaTail):
         agent = self.agent

      cost = agent.get_cost(action)
      agent.food -= cost[0]
      agent.oxy -= cost[1]

   def do_action2(self, move):
      agent = self

      if move.location == 1:
         agent = self.get_agent(1)
      elif move.location == 0:
         agent = self.get_agent(0)

      if not agent == self:
         agent.do_action2(move)
         return

      pass

   def do_action(self, action, cell, location, agent=None):

      #give content with the move

      #just give this the move

      success = False
      #direction = False
      #if location == 0:
      #direction = True

      if not agent == None:
         #then go to that agent to start
         agent.do_action(action, cell, location)
         return

      #agent_pos = self.pos
      #if location == 1:
      #   agent_pos = self.__get_tail_pos()

      #send content of cell and agent pos

      if action == "move":  # ACCOUNT FOR BACKWARDS MOVEMENT???

         #if height does not chnage then simply shift the head

         testagent = self

         if location == 1:
            testagent = self.get_agent(location)

         testagent.stage(cell, location)
         testagent.execute(location)
         pass

      elif action == "action:eat" and location == 0:
         contents = self.cell_contents(cell)
         if len(contents) >= 1:  # Need to deal with contents not being a list
            for agent in contents:
               if isinstance(agent, Food):
                  self.eat(agent)
                  success = True
                  if agent.amount <= 0:
                     self.stage(cell, location)
                     self.execute(location)
                  break

      elif action == "action:breathe":
         if location == 1:
            self.breathe()
            success = True

      elif action == "action:breathe+move":
         if location == 1:
            self.stage(cell, location)
            self.execute(location)
            self.breathe()
            success = True

      elif action == "wait":
         success = True
         pass

      if success:
         self.apply_cost(action)

   def die(self):
      #print(str(self), "running self.die")
      if not self.next == None:
         self.next.die()

      #overriding the overriden method
      #ExtendedAgent.die()
      #print("DIEDIE")
      if self.kill_flag == False:
         self.kill_flag = True
         self.model.kill_agents.append(self)
         return True

      return False

   def __check_conditions(self):
      if self.food <= 0 or self.oxy <= 0:
         return True

      return False



   def step(self):
      if self.__check_conditions():
         self.die()
         return
      
      if self.poop > self.POOP_THRESHOLD:
         self.create_poop(self.poop)
         self.poop = 0

      neighborhood_object = self.get_neighborhood(self.pos,center=False,moore=True)
      pos = None
      move = None

      if self.oxy <= 3.0:
         #Low oxy
         #change to returing breathible rather than storing it?
         oxy_moves = neighborhood_object.get_breatheable()#breatheable_list
         if len(oxy_moves) > 0:
            self.breathe()
            self.food -= 0.1
            return

         else:
            #if we are here then there are no empty spaces above the agent so we must try and swap positions
            swap_moves = neighborhood_object.get_swapable(direction="up")
            if len(swap_moves) > 0:
               #choose one randomly and start eating
               index = self.randindex(len(swap_moves))
               #pos = swap_moves[index].pos
               cellObj = swap_moves[index]
               #print(str(cellObj))
               if cellObj.Poop is not None:
                  self.swap_position(cellObj.Poop)
               elif cellObj.Food is not None:
                  self.swap_position(cellObj.Food)
               self.oxy -= 0.3
               self.food -= 0.3
               return
            pass

      else:
         #try to eat
         if not self.eating is None and self.eating.amount > 0:
            self.eat(self.eating)
            #need to also do a check if eating is in the neighborhood ****************************
            #self.oxy -=1 #masybe remove this to have it simulate enzymes
            return

         else:
            eat_moves = neighborhood_object.get_edible()
            if len(eat_moves) > 0:
               #choose one randomly and start eating
               index = self.randindex(len(eat_moves))
               food_agent = eat_moves[index].Food
               self.eating = food_agent
               self.eat(food_agent)
               self.oxy -= 1
               return
            else:
               #try and go down
               down_moves = neighborhood_object.get_moveable(direction="down")

               if len(down_moves) > 0:
                  index = self.randindex(len(down_moves))
                  pos = down_moves[index].pos
                  self.oxy -= 0.1
                  self.food -= 0.1

               else:
                  swap_moves = neighborhood_object.get_swapable(direction="down")
                  if len(swap_moves) > 0:
                     index = self.randindex(len(swap_moves))
                     cellObj = swap_moves[index]
                     print(str(cellObj))
                     if cellObj.Poop is not None:
                        self.swap_position(cellObj.Poop)
                     elif cellObj.Food is not None:
                        self.swap_position(cellObj.Food)

                     self.oxy -= 0.2
                     self.food -= 0.2
                     return
                  else:
                     pass

               
      if (not pos == None) and (not self.model.grid.out_of_bounds(pos)):
         move = pos

      if move is None or self.is_solid(move):
         move_list = neighborhood_object.get_moveable()
         if len(move_list) > 0:
            index = self.randindex(len(move_list))
            move = move_list[index].pos
            self.oxy -= 0.1
            self.food -= 0.1

      self.move_to(move)





   def stepbac(self):
      #first check if death conditions are valid, if so add to kill list and early return
      self.update_values()
      #print(self.BREATHE_AMOUNT)

      #print(self.food)
      #print(self.oxy)


      if self.__check_conditions():
         self.die()
         return

      if self.poop > self.POOP_THRESHOLD:
         self.create_poop(self.poop)
         self.poop = 0

      neighbors = self.neighbors()#self.neighbor_contents()
      contents = self.neighbor_contents(neighbors = neighbors)
      move_list = list()


      for pos,content in zip(neighbors,contents):
         move_list.append(PotentialMove(pos,content))
         #if isinstance(neighbor,list()):
            #for element in neighbor:

      #if self.oxy < self.LOW_OXY:
         

      #index = random.randint(0, len(move_list) - 1)
      #move = move_list[index]

      #self.move_to(move)

      #Now search through the list according to parameters
      move = None
      pos = None


      if self.oxy < 3.0:
         #print("LowO")
         #pos = (self.pos[0], self.pos[1] + 1)
         neighbors = self.neighbors(moore=False)
         contents = self.neighbor_contents(neighbors = neighbors)
         #check that there is oxygen to breathe

         contains_oxy = False
         for neighbor,contents in zip(neighbors,contents):
            if contains_oxy is True:
               break
            if self.is_solid(neighbor):
               continue
            if isinstance(contents, list):
               for element in contents:
                  if isinstance(element,Oxygen):
                     contains_oxy = True
                     break
            if isinstance(contents, Oxygen):
               contains_oxy = True
               break

         if contains_oxy:
            self.add_oxy(self.BREATHE_AMOUNT)
            self.food -= 1
            pos = "OXY"
         else:
            # move = None
            ppos = (self.pos[0], self.pos[1] + 1)
            print("trying to move up")
            if self.is_food(ppos):  # self.is_solid(ppos)
              #move up
               food_agent = self.get_food(ppos)
               self.swap_position(food_agent)
               self.oxy -=0.3
               self.food -= 0.3
               pos = "OXY_Move"
            if self.is_poop(ppos):  # self.is_solid(ppos)
              #move up
               poop_agent = self.get_poop(ppos)
               self.swap_position(poop_agent)
               self.oxy -=0.3
               self.food -= 0.3
               pos = "OXY_Move"




      else:
         #if eating is not possible try and go downd

         if not self.eating is None and self.eating.amount > 0:
            self.eat(self.eating)
            pos = "Food"

         else:
            neighbors = self.neighbors(moore=False)

            food_list = list()
            poop_list = list()

            for neighbor in neighbors:
               #if self.is_food(neighbor)
               food_agent = self.get_food(neighbor)

               if food_agent is not None:
                  if not food_agent.is_poop:
                     food_list.append((neighbor, food_agent))
                  else:
                     #poop_list.append((neighbor,food_agent))
                     pass

            if len(food_list) > 0:
               index = random.randint(0, len(food_list)-1)
               food_agent = food_list[index][1]
               self.eating = food_agent
               self.eat(food_agent)
               self.oxy -= 1
               pos = "Food"
            elif len(poop_list) > 0:
               index = random.randint(0, len(poop_list)-1)
               food_agent = poop_list[index][1]
               self.eating = food_agent
               self.eat(food_agent)
               self.oxy -= 1
               pos = "Food"
            else:
               #move = None
               pos = (self.pos[0], self.pos[1] - 1)



         #print("FOOOOOD")
         #pos = (self.pos[0], self.pos[1] - 1)
         #food_agent = self.get_food(pos)

         #if not food_agent is None:
            #print("Eating")
            #self.eat(food_agent)
            #self.oxy -= 1
            #pos = "Food"
         #self.food += 1.2

      """else:

         if not self.eating is None and self.eating.amount > 0:
            self.eat(self.eating)

         else:
            neighbors = self.neighbors(moore=False)

            food_list = list()

            for neighbor in neighbors:
               #if self.is_food(neighbor)
               food_agent = self.get_food(neighbor)

               if food_agent is not None:
                  food_list.append((neighbor,food_agent))

            if len(food_list) > 0:
               index = random.randint(0, len(food_list))
               food_agent = food_list[index]
               self.eating = food_agent
               self.eat(food_agent)
               self.oxy -= 1
            else:
               move = None
      """
      if isinstance(pos,str):
         print("POS is str")
         print(pos)
         return

      if (not pos == None) and (not self.model.grid.out_of_bounds(pos)):
         move = pos

      if move is None or self.is_solid(move):
         index = random.randint(0, len(move_list) - 1)
         move = move_list[index].pos
         self.oxy -= 0.1
         self.food -= 0.1
            
      self.move_to(move)



      """
      if self.oxy < 3.0:
         #go up (look for moves with a positive reward for oxy)
         pos = (self.pos[0], self.pos[1] + 1)
         self.oxy += 1.2
         self.food -= 1


      #elif self.food < 3.0:
      else:
         pos = (self.pos[0], self.pos[1] - 1)
         food_agent = self.get_food(pos)

         if not food_agent is None:
            self.eat(food_agent)
            self.oxy -= 1
         #self.food += 1.2


      if (not pos == None) and (not self.model.grid.out_of_bounds(pos)):
         move = pos

      if move == None or self.is_solid(move):
         index = random.randint(0, len(move_list) - 1)
         move = move_list[index].pos
         self.oxy -= 0.1
         self.food -= 0.1

      self.move_to(move)
         """

   def swap_position(self,agent):
      pos1 = self.pos
      pos2 = agent.pos

      self.move_to(pos2,suppress_restrictions=True)
      agent.move_to(pos1)


   def get_agent_pos(self, location=None):
      return self.get_agent(location).pos

   def get_agent(self, location=None):
      agent = self

      if (location == 0 or location == None) and not agent.previous == None:
         while not agent.previous == None:
            agent = agent.previous

      elif location == 1 and not agent.next == None:
         while not agent.next == None:
            agent = agent.next

      return agent

   def move_to(self, pos, suppress_restrictions=False):

      if suppress_restrictions == True:
         #self.update_facing(pos)
         self.model.grid.move_agent(self, pos)
         return True

      if pos is not None:

         if self.model.grid.out_of_bounds(pos):
            print("Out of bounds")
            return False
         if self.is_solid(pos):
            print("Attempted to move into a solid block")
            return False
         else:
            self.record_move(pos)
            #self.update_facing(pos)
            self.model.grid.move_agent(self, pos)
            return True

   def record_move(self,pos):
      #print("Recording move")
      content = self.cell_contents(pos)
      if len(content) >= 1:
         content = content[0]
      #print(str(content))
      if isinstance(content,Oxygen):
         #print("Is instance")
         content.enter_count += 1
         #print(content.enter_count)


   def eat(self, food_agent):

      amount = self.EAT_AMOUNT
      consumed_amount = food_agent.consume(amount)
      self.add_food(consumed_amount * self.EAT_EFFICIENCY)
      #create a food poop of size EAT_AMOUNT - amount
      self.poop += consumed_amount * (1-self.EAT_EFFICIENCY)
      #self.create_poop(consumed_amount * (1 - self.EAT_EFFICIENCY))


   def create_poop(self,amount):
      #change this to depend on where the agent is facing
      pos = (self.pos[0],self.pos[1] + 1)

      food = self.get_food(pos)

      if food is not None:
         food.amount += amount
         return

      id = self.model.next_id()

      #agent = Food(id,self.model,amount=amount)
      agent = Poop(id,self.model,amount=amount)
      #agent.check_for_update = True
      #agent.is_poop = True
      self.model.schedule.add(agent)
      self.model.grid.place_agent(agent,pos)
      agent.fall_sideways()

      #print(str(agent))

      self.times_pooped += 1


class LarvaTail(LarvaAgent):
   #  Agent - Tail - Tail - Tail
   def __init__(self, unique_id: int, model: Model, type, main_agent, previous=None, next=None, end=False, pos=None):
      #super().__init__(unique_id, model, type)
      self.unique_id = unique_id
      self.model = model
      self.staged_move = None
      self.agent = main_agent
      self.previous = previous
      self.next = next
      self.end = end

      self.kill_flag = False

   def step(self):
      pass

   def get_pos(self):
      return self.pos

   def is_end(self):
      return self.end

   def die(self):
      if not self.next == None:
         self.next.die()
      if not self.pos == None:
         self.model.grid.remove_agent(self)
      del self


class Food(ExtendedAgent):
   def __init__(self, unique_id: int, model: Model, pos=None, amount = None):
      super().__init__(unique_id, model)
      self.amount = 10 #2
      if not amount is None:
         self.amount = amount
      self.is_food = True
      #self.is_poop = False
      self.kill_flag = False
      self.check_for_update = False

   def step(self):
      if self.amount <= 0:
         self.die()
         self.updateNeighbors()

      if (not self.check_for_update) and (not self.kill_flag):

         self.updatePos()
         self.fall_sideways()
         self.check_for_update = False
      #solid
      #if isinstance(content,list()):
      #   for element in content:
      #      if isinstance(element,(Food,LarvaAgent,LarvaTail)):

   def move_to(self,pos):
      self.updateNeighbors()
      self.model.grid.move_agent(self, pos)


   def getFood(self,content):
      if isinstance(content,list):
         for element in content:
            if isinstance(element,Food):
               return (True,element)
      else:
         if isinstance(content,Food):
            return (True, content)

      return (False,None)


   def updateNeighbors(self):
      #change this to check for above food agent as well as left and right +2 agents
      #neighbors = [self.getFood(n)[1] for n in self.neighbors() if self.getFood(n)[0]]

      pos_above = (self.pos[0], self.pos[1] + 1)
      pos_left2 = (self.pos[0] - 1, self.pos[1] + 2)
      pos_right2 = (self.pos[0] + 1, self.pos[1] + 2)

      def change_update_flag(self,pos):
         if not self.model.grid.out_of_bounds(pos):
            res = self.getFood(pos)
            if res[0]:
               res[1].check_for_update = True
         return

      change_update_flag(self,pos_above)
      change_update_flag(self, pos_left2)
      change_update_flag(self, pos_right2)

   def get_above_food(self):
      pos = (self.pos[0], self.pos[1] + 1)
      content = self.cell_contents(pos)
      agent = None
      if not self.model.grid.out_of_bounds(pos):
         if isinstance(content, list):
            for element in content:
               if isinstance(element,Food):
                  agent = element
                  break
      #print(str(agent))
      return agent


   def updatePos(self,do_checks = None):
      #Run when a tunnel collapses and all above food agents need to fall
      pos = (self.pos[0], self.pos[1] - 1)

      if do_checks == False or (not self.is_solid(pos) and not self.model.grid.out_of_bounds(pos)):
         food_agent = self.get_above_food()
         self.move_to(pos)
         if not food_agent is None:
            food_agent.updatePos(do_checks = False)
      
   def fall_sideways(self):

      left_pos_1 = None
      left_pos_2 = None

      if self.pos[0] >= 1 and self.pos[1] >= 2:
         left_pos_1 = (self.pos[0] - 1, self.pos[1] - 1)
         left_pos_2 = (self.pos[0] - 1, self.pos[1] - 2)

      right_pos_1 = None
      right_pos_2 = None

      if self.pos[0] < self.model.grid.width - 1 and self.pos[1] >= 2:
         right_pos_1 = (self.pos[0] + 1, self.pos[1] - 1)
         right_pos_2 = (self.pos[0] + 1, self.pos[1] - 2)

      if all(p is None for p in [left_pos_1,left_pos_2,right_pos_1,right_pos_2]):
         return

      unstable_left = False
      unstable_right = False

      if all([left_pos_1,left_pos_2]) and (not self.is_solid(left_pos_1)) and (not self.is_solid(left_pos_2)):
         unstable_left = True
      
      if all([right_pos_1,right_pos_2]) and (not self.is_solid(right_pos_1)) and (not self.is_solid(right_pos_2)):
         unstable_right = True

      if (not unstable_left) and (not unstable_right):
         return

      if unstable_left and unstable_right:
         rand = random.randint(0, 1)
         if rand == 0:
            self.move_to(left_pos_2)
         else:
            self.move_to(right_pos_2)
      elif unstable_left:

         self.move_to(left_pos_2)
      elif unstable_right:
         self.move_to(right_pos_2)
      
      self.fall()

   def fall(self):
      #fall as many blocks as possible until reaching a solid block

      pos = (self.pos[0], self.pos[1] - 1)

      def condition(self,pos):
         if (not self.is_solid(pos)) and not self.model.grid.out_of_bounds(pos):
            return True
         return False

      while condition(self,pos):
         self.move_to(pos)
         pos = (pos[0], pos[1] - 1)


   def checkIfStable(self):
      pos = (self.pos[0], self.pos[1] - 1)

      pass


   def consume(self, amount):
      amt = 0
      amount = abs(amount)
      if self.kill_flag == False:
         if self.amount - amount <= 0:
            amt = self.amount
            self.amount = 0
            #self.die()
         else:
            self.amount -= amount
            amt = amount

      return amt


class Oxygen(ExtendedAgent):
   def __init__(self, unique_id: int, model: Model, depleted=None, traversable=None, pos=None):
      super().__init__(unique_id, model)
      if depleted == None:
         self.depleted = False
      else:
         self.depleted = depleted
      if traversable == None:
         self.traversable = False
      else:
         self.traversable = traversable

      self.surrounded = False

   def update(self):
      #Update oxy state based on surroundings
      neighbors = self.neighbors(center=True, moore=False)
      UNDEPLETE = 1
      count = 0

      #problem need to do different neightbors for oxy and

      for cell in neighbors:
         contents = self.model.grid.get_cell_list_contents(cell)
         if isinstance(contents, list) and len(contents) > 1:
            temp_var = False
            for item in contents:
               if isinstance(item, Food):  # LarvaAgent)): #LarvaTail)):
                  temp_var = True
            self.traversable = temp_var
         else:
            item = contents[0]
            if isinstance(item, Oxygen) and item.depleted == False:
                count += 1

      if count >= UNDEPLETE:
         self.depleted = False

      pass

   def get_traversable(self):
      #self.update()
      return self.traversable

   def step(self):
      #self.update()
      pass



class Poop(Food): #extended agnet or food??
   def __init__(self,unique_id: int, model: Model, pos = None, amount = None):
      super().__init__(unique_id, model, amount = amount)
      self.SPREAD_THRESHOLD = 10
      self.is_food = False
      self.is_poop = True
      pass

   def find_empty_pos(self):
      #neighbors = self.neighbors()
      #contents = self.neighbor_contents(neighbors=neighbors)
      #for neighbor,contents in zip(neighbors,contents):

      #check below first, then check sides, then above

      def check_if_empty(contents):
         bool = 0
         if isinstance(contents, (Food, LarvaAgent, LarvaTail)):
            return 0
         if isinstance(contents,Oxygen):
            return 1
         for content in contents:
            #is solid
            if isinstance(content,(Food,LarvaAgent,LarvaTail)):
               bool = 0
               return bool
            if isinstance(content,Poop):
               bool = 2
         
         return bool



      pos1 = (self.pos[0], self.pos[1] - 1)
      pos2 = (self.pos[0] + 1, self.pos[1])
      pos3 = (self.pos[0] - 1, self.pos[1])
      pos4 = (self.pos[0], self.pos[1] + 1)
      #contents = self.
      if random.randint(0,1):
         temp = pos2
         pos2 = pos3
         pos3 = temp

      pos_list = [pos1,pos2,pos3,pos4]
      for pos in pos_list:
         if check_if_empty(self.cell_contents(pos)) == 1:
            return pos
      #for pos in pos_list:
      #   if check_if_empty(self.cell_contents(pos)) == 2:
      #      return pos
      



   """def spread(self):
      #check neighbors for open slots and create an agent in an open space with half the amount
      pos = self.find_empty_pos()

      if pos is not None:
         amount = self.amount / 2
         agent = Poop(self.model.next_id(),self.model, amount)
         self.model.schedule.add(agent)
         self.model.grid.place_agent(agent,pos)
         self.amount *= 0.5

      pass
   """

   #spread it left and right
   def spread(self):
      pos1 = (self.pos[0] - 1, self.pos[1])
      pos2 = (self.pos[0] + 1, self.pos[1])
      
      if self.model.grid.out_of_bounds(pos1):
         pos1 = None
      if self.model.grid.out_of_bounds(pos2):
         pos2= None

      left_solid = self.is_solid(pos1)
      right_solid = self.is_solid(pos2)

      left_poop = self.get_poop(pos1)
      right_poop = self.get_poop(pos2)

      update_list = list()
      amt = 0
      if (not left_solid) and (not right_solid):
         amt = round(self.amount/3,1)
         update_list.append((pos1,left_poop))
         update_list.append((pos2, right_poop))

      elif not left_solid:
         amt = round(self.amount/2,1)
         update_list.append((pos1, left_poop))
      elif not right_solid:
         amt = round(self.amount/2,1)
         update_list.append((pos2, right_poop))

      for cell in update_list:
         if cell[1] is None:
            #create agent
            self.create_agent(cell[0],amt)
            pass
         else:
            #add to agent
            cell[1].add_amount(amt)
            pass
      
      self.amount = amt

   #just call create agent at the two popsitions and have it check, add to agentr if one exits and do nothing if solid??

   def fall_sideways(self):

      left_pos_1 = None

      if self.pos[0] >= 1 and self.pos[1] > 0:
         left_pos_1 = (self.pos[0] - 1, self.pos[1] - 1)

      right_pos_1 = None

      if self.pos[0] < self.model.grid.width - 1 and self.pos[1] > 0:
         right_pos_1 = (self.pos[0] + 1, self.pos[1] - 1)

      if all(p is None for p in [left_pos_1, right_pos_1]):
         return

      unstable_left = False
      unstable_right = False

      if all([left_pos_1]) and (not self.is_solid(left_pos_1)):
         unstable_left = True

      if all([right_pos_1]) and (not self.is_solid(right_pos_1)):
         unstable_right = True

      if (not unstable_left) and (not unstable_right):
         return

      if unstable_left and unstable_right:
         rand = random.randint(0, 1)
         if rand == 0:
            self.move_to(left_pos_1)
         else:
            self.move_to(right_pos_1)
      elif unstable_left:

         self.move_to(left_pos_1)
      elif unstable_right:
         self.move_to(right_pos_1)

      self.fall()



   def create_agent(self,pos,amount):
      id = self.model.next_id()
      agent = Poop(id,self.model,amount=amount)
      self.model.schedule.add(agent)
      self.model.grid.place_agent(agent,pos)

   #def __spread_agent(self,pos,amount):
      #if not self.is_solid(pos):


   def diffuse(self,agent):
      diff = 0
      if self.amount > 0 and agent.amount > 0:
         diff = (self.amount - agent.amount)/2
      else:
         return
      
      print(str(diff))
      agent.add_amount(diff)
      self.amount -= diff


   def diffuse_neighborhood(self):
      pass

   def add_amount(self,amount):
      self.check_for_update = True
      self.amount += amount

   #def fall(self):
      #pass
   def step(self):
     # print("running step on poop")
      #print("Amount is: ", str(self.amount))
      if self.amount <= 0:
         #print("POOP dying")
         self.die()
      if self.amount > self.SPREAD_THRESHOLD:
         self.spread()

      self.fall()
      self.fall_sideways()

      if self.check_for_update:
         self.amount = round(self.amount, 1)
         self.diffuse()





class LarvaModel(Model):

   def __init__(self, N, width, height, **kwargs):
   
   #breathe_amount, eat_amt, eat_efficiency):#larva_params):
      FOOD_HEIGHT = height - (4+1)

      #print("init model")
      super().__init__()
      self.num_agents = N
      self.schedule = RandomActivation(self)
      self.grid = MultiGrid(width, height, False)
      self.kill_agents = list()

      self.larva_params = kwargs #LarvaParams(kwargs)
      
      #kwargs #Params(breathe_amount, eat_amt, eat_efficiency, None, None, None, None)


      self.create_agents(width, height)


   def create_agents(self, width, height):
      FOOD_HEIGHT = height - (4+1)
      #create agents
      for i in range(self.num_agents):
         agent = LarvaAgent(self.next_id(), self, "test")
         self.schedule.add(agent)

         #2(i-1)

         x = math.floor(i*(width-1)/self.num_agents)

         #x = 2*i
         y = math.floor(FOOD_HEIGHT)
         self.grid.place_agent(agent, (x, y))
         i = 0
         while (not agent.next == None):
            i += 1
            agent = agent.next
            #self.schedule.add(agent)
            self.grid.place_agent(agent, (x, y+i))

      for i in range(FOOD_HEIGHT):
         for j in range(width):
            id = self.next_id()
            agent = Food(id, self)
            #print(str(agent))
            self.schedule.add(agent)
            self.grid.place_agent(agent, (j, i))


      for i in range(height):
         for j in range(width):
            id = self.next_id()
            depleted = True
            traversable = True
            agent = Oxygen(id, self, depleted, traversable)
            self.schedule.add(agent)
            self.grid.place_agent(agent, (j, i))

      agent = Poop(self.next_id(),self,amount=100)
      self.schedule.add(agent)
      self.grid.place_agent(agent, (width - 1, height - 1))

   def step(self):
      '''Advance the model by one step.'''
      self.schedule.step()
      for x in self.kill_agents:
         #print(self.kill_agents)
         #print(x)
         #print(x.pos)
         self.kill_agents.remove(x)
         self.grid.remove_agent(x)
         self.schedule.remove(x)
