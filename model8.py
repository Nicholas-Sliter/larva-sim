import math
import random
from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from utils import DoublyLinkedList
from collections import namedtuple

LARVA_SIZE = 4
EAT_AMOUNT = 1.5


Point = namedtuple('Point', 'x y')
Cost = namedtuple('Cost', ['food', 'oxygen'])
Reward = namedtuple('Reward', ['food', 'oxygen'])
Move = namedtuple("Move", ["pos", "action", "location", "cost", "reward", "value"])
RankedMove = namedtuple("Rank", ["move", "rank"])



class ExtendedAgent(Agent):
   def __init__(self,unique_id: int, model: Model):
      super().__init__(unique_id, model)
      self.kill_flag = False
      self.type = None

   def get_position(self):
      return self.pos

   def neighbors(self, pos=None, center=False, moore=True):

      if (pos == None):
         pos = self.pos

      if isinstance(pos,(ExtendedAgent,LarvaAgent,Agent,LarvaTail,Oxygen,Food)):
         pos = pos.pos

      neighbors = self.model.grid.get_neighborhood(
          pos,
          moore=True,
          include_center=False)

      return neighbors

   def neighbor_contents(self, pos=None, neighbors=None):
      if pos == None and neighbors == None:
         return list()
      if pos == None:
         pos = self.pos
      if neighbors == None:
         neighbors = self.neighbors(pos)

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

   def die(self):
      print(str(self), "running self.die")
      if self.kill_flag == False:
         self.kill_flag = True
         self.model.kill_agents.append(self)
         return True
      
      return False



class LarvaAgent(ExtendedAgent):

   def __init__(self, unique_id: int, model: Model, type=None, next=None, pos=None):
      super().__init__(unique_id, model)
      self.type = type
      self.food = 2.0
      self.oxy = 3.0
      self.staged_move = None

      self.end = False
      self.previous = None
      self.next = None

      #create tail agents here
      previous_agent = self
      for i in range(LARVA_SIZE-1):
         endv = False
         if i+2 == LARVA_SIZE:
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
               self.next.stage(move,location)
               #self.staged_move = cell

         else:
            if not self.previous == None:
               #self.staged_move = cell
               move = self.pos
               self.previous.stage(move,location)
               #self.staged_move = cell
         
         self.staged_move = cell
      """
      Stages changes for tail agents so that they can be executed at once and move with the head without detachment

      """

   def execute(self, direction=0):
      #print("execute running with: ", self.pos,"direction = ",direction)
      #print("staged_move is: " + str(self.staged_move))

      cell = self.staged_move
      print("executing move from",self.pos,"to",cell)

      #if isinstance(self, LarvaTail) and self.end == False:
         #self.move_to(cell)  # ,suppress_restrictions=True)
      #else:
      self.move_to(cell,suppress_restrictions=True)

      if direction == 0 and (not self.next == None):
         self.next.execute(direction)
      elif direction == 1 and (not self.previous == None):
         self.previous.execute(direction)

   def neighbors(self, pos=None):
      if (pos == None):
         pos = self.pos
      neighbors = self.model.grid.get_neighborhood(
          pos,
          moore=True,
          include_center=False)
      return neighbors

   def neighbor_contents(self, pos=None, neighbors=None):
      if neighbors == None:
         neighbors = self.neighbors(pos)
      return self.model.grid.get_cell_list_contents(neighbors)

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
      BREATHE_AMT = 1.5

      #action = move.action
      reward_food = 0
      reward_oxy = 0

      if action == "move":
         reward_food = 0
         reward_oxy = 0

      elif action == "action:breathe" or action == "action:breathe+move":
         reward_food = 0
         reward_oxy = BREATHE_AMT

      elif action == "action:eat":
         reward_food = min(content.amount,EAT_AMOUNT) + 2
         reward_oxy = 0

      elif action == "wait":
         reward_food = 0
         reward_oxy = 0

      return Reward(reward_food,reward_oxy)

   def get_possible_steps(self, location=None):
      #first find the pos to use based on location.
      # location == None or location == 0 imply using self.pos
      # location == 1 should use the tail pos
      pos = None
      modifier = ""

      if location == None or location == 0:
         agent_pos = self.pos
         modifier = "head"
      else:
         agent = self
         modifier = "tail"
         while not agent.next == None:
            #Returns the tail as agent
            agent = agent.next
         agent_pos = agent.pos
         print("Agent POS is: " + str(agent_pos))

      #Now get the neighborhood around pos
      neighbors = self.neighbors(agent_pos)
      #print("Neighbors are: " + str(neighbors))
      neighbors_contents = self.neighbor_contents(neighbors=neighbors)
      #print("Neighbor contents are: " + str(neighbors_contents))
      possible_steps = list()

      #["pos", "action", "location", "cost", "reward", "value"])
      def create_move(self,content,pos,action,location):
         #print("create_move: starting with: " + str(content) + str(pos) + str(action) + str(location))
         #find cost, reward, and value
         cost = self.get_cost(action,location)
         #print("create_move: cost is: " + str(cost))
         reward = self.get_reward(content,action)
         #print(reward)
         value = self.__get_value(cost,reward)

         move = Move(pos,action,location,cost,reward,value)

         return move
      def is_actionable(self,pos,content):
         #finds if a given cell pos is actionable #return actioanble list??
         actionable = False

         def guard_conditions(self,pos,content):
            def conditions(self,pos,content):
               if (content == None or self.model.grid.is_cell_empty(pos)):
                  return False
               if isinstance(content, (LarvaAgent, LarvaTail)):
                  return False
               #if isinstance(content, Oxygen) and content.traversable == False:
                 # return False
               
               else:

                  return True

            if isinstance(content,list):
               actionable = True
               for element in content:
                  if conditions(self,pos,element) == False:
                     actionable = False

               return actionable
            else:
               return conditions(self,pos,content)
               

         #return tuple(actionable, list of actionable contents)
         #Guard conditions
         if guard_conditions(self,pos,content) == False:
            #print('Guard conditions triggered')
            return False
         #if isinstance(content, Oxygen) and content.traversable == False:
         #   print('Guard conditions triggered: non-traversable oxygen')
         #   return False

         else:

            return True

      def get_action(self,pos,content,location,agent_pos):
         action_list = list()

         #print("get_action:", pos, content, location, agent_pos)

         if isinstance(content, Oxygen) and content.get_traversable():
            action = "move"
            #move = Move(pos, action, location, self.get_cost(action))
            #temp_list.append(move)
            action_list.append(action)

            if location == 1 and content.depleted == False and pos[1] >= agent_pos[1]:  # append another for breathing at tail
               #cost = None
               action = "action:breathe"
               if not pos == agent_pos:
                  action = "action:breathe+move"
               action_list.append(action)
               
         if isinstance(content, Food) and location == 0 and pos[1] <= agent_pos[1]:
            #print("FOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOD!!")
            action = "action:eat"
            action_list.append(action)

         return action_list

      move_list = list()

      for cell, contents in zip(neighbors,neighbors_contents):
         #print(cell,contents)
         if is_actionable(self, cell, contents):
            #print(str(cell) + " is actionable")
         #go through each elements in contents and create a move for it

            if isinstance(contents,list):
               for element in contents:
                  action_list = get_action(self,cell,element,location,agent_pos)

                  if (not action_list == None) and len(action_list) >= 1:
                     for action in action_list:
                        move = create_move(self,element,cell,action,location)
                        move_list.append(move)
            else:
               action_list = get_action(self, cell, contents, location, agent_pos)
               if (not action_list == None) and len(action_list) >= 1:
                  for action in action_list:
                     move = create_move(self, contents, cell, action, location)
                     move_list.append(move)


         if not move_list == None:
            possible_steps.extend(move_list)
            #possible_steps.append(cell)

      return move_list

   def __get_value(self, cost, reward):
      def greedy_func(move_cost, move_reward, current_amount, buffer):
         food_cost = move_cost[0]
         food_reward = move_reward[0]
         food_amount = current_amount[0]
         food_buffer = buffer[0]

         #food_value = (food_reward + food_buffer) / \
         #    (food_amount - food_cost - food_buffer)

         food_value = (food_amount + (food_reward - food_cost) ) / food_amount

         oxy_cost = move_cost[1]
         oxy_reward = move_reward[1]
         oxy_amount = current_amount[1]
         oxy_buffer = buffer[1]

         #oxy_value = (oxy_reward + oxy_buffer) / \
         #    (oxy_amount - oxy_cost - oxy_buffer)

         oxy_value = (oxy_amount + (oxy_reward - oxy_cost)) / oxy_amount

         #maybe return the min of the two?

         value = round((food_value + oxy_value)/2, 4)

         #value = round

         return value

      current_amount = (self.food, self.oxy)
      buffer = (0.2, 0.2)

      return greedy_func(cost, reward, current_amount, buffer)

   def __greedy_cost(self, action, reward):
      def greedy_func(move_cost, move_reward, current_amount, buffer):
         food_cost = move_cost[0]
         food_reward = move_reward[0]
         food_amount = current_amount[0]
         food_buffer = buffer[0]

         food_value = (food_reward + food_buffer) / \
             (food_amount - food_cost - food_buffer)

         oxy_cost = move_cost[1]
         oxy_reward = move_reward[1]
         oxy_amount = current_amount[1]
         oxy_buffer = buffer[1]

         oxy_value = (oxy_reward + oxy_buffer) / \
             (oxy_amount - oxy_cost - oxy_buffer)

         #maybe return the min of the two?
         return food_value + oxy_value

      cost = self.get_cost(action)
      current_amount = (self.food, self.oxy)
      buffer = (0.2, 0.2)

      return greedy_func(cost, reward, current_amount, buffer)

   def rank(self, head_moves=None, tail_moves=None):
      move_list = list()

      if head_moves == None and tail_moves == None:
         return None

      if not tail_moves == None:
         for move in tail_moves:
            #x = RankedMove(move,rank)
            move_list.append(move)

      if not head_moves == None:
         for move in head_moves:
            move_list.append(move)

      move_list.sort(reverse=True, key=lambda move: move.value)
      print("Move list is:",str(move_list))

      return move_list

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

      if location == None: location = 0

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
         OXY_COST = 0.1 + 0.5 +1.0
         FOOD_COST = 0.2 + 1.0 +1.0
      elif action == "wait":
         OXY_COST = 0.1
         FOOD_COST = 0.1

      if location == 1:
         OXY_COST *= 1.2
         FOOD_COST *= 1.2
      elif action == "move" and location == 0:
         OXY_COST = 0.1
         FOOD_COST = 0.1
      return Cost(OXY_COST, FOOD_COST)

   def breathe(self):
      agent = self
      if isinstance(self,LarvaTail):
         agent = self.agent
      INCR_AMT = 2
      agent.oxy += INCR_AMT
      print("Oxy for", str(agent.unique_id), "is",str(agent.oxy))

   def get_movement_cost(self):
      #gets the cost to move from the current position to pos
      OXY_COST = 0.5
      FOOD_COST = 1

      return (OXY_COST, FOOD_COST)

   def apply_cost(self, action):
      agent = self
      if isinstance(self,LarvaTail):
         agent = self.agent

      cost = agent.get_cost(action)
      agent.food -= cost[0]
      agent.oxy -= cost[1]

   def do_action(self, action, cell, location, agent = None):

      #give content with the move


      success = False
      #direction = False
      #if location == 0:
         #direction = True

      if not agent == None:
         #then go to that agent to start
         agent.do_action(action,cell,location)
         return


      #agent_pos = self.pos
      #if location == 1:
      #   agent_pos = self.__get_tail_pos()


      #send content of cell and agent pos 


      if action == "move": #ACCOUNT FOR BACKWARDS MOVEMENT???
         testagent = self

         if location == 1: 
            testagent = self.get_agent(location)

         testagent.stage(cell, location)
         testagent.execute(location)
         pass

      elif action == "action:eat" and location == 0:
         print("###########Runnign Action EAT############")
         contents = self.cell_contents(cell)
         if len(contents) >= 1: #Need to deal with contents not being a list
            for agent in contents:
               print("###############EAT################ CHECKING AGENT")
               print(str(agent))
               if isinstance(agent, Food):
                  print("Isinstance food")
                  self.eat(agent)
                  success = True
                  if agent.amount <= 0:
                     self.stage(cell,location)
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


   def __check_conditions(self):
      if self.food <= 0 or self.oxy <= 0:
         return True

      return False

   def step(self):
      #first check if death conditions are valid, if so add to kill list and early return

      print("Starting step")


      if self.__check_conditions():
         self.die()
         return


      # find valid moves at head and tail, rank those moves, choose the best move, then move, do an action, or do nothing
      possible_steps_head = self.get_possible_steps(0)
      possible_steps_tail = self.get_possible_steps(1)
      #print(possible_steps_head)
      #print(possible_steps_tail)

      ranked_moves = self.rank(possible_steps_head, possible_steps_tail)
      #each move should have a positional argument as well as a type argument
      # position = cell (x,y) and type = one of "move", "action:eat", "action:breathe", "wait"

      move = None
      value = float('-inf')
      index = 0
      if len(ranked_moves) > 1:
         #move = ranked_moves[0]
         while index < len(ranked_moves): #-1 ???
            mv = ranked_moves[index]
            value = max(value,mv.value)
            if value == mv.value:
               index += 1
            else:
               break

      index = random.randint(0,index)
      print(index)
      if index < len(ranked_moves):
         move = ranked_moves[index]
      print(move)

      if not (move == None):
         #if move is action then do it, if it is move then stage and execute move, else do nothing
         #and apply costs
         pos = move.pos
         action = move.action
         location = move.location
         reward = move.reward

         #print("The selected move for the agent with id", str(self.unique_id), "and position",self.pos,"is",str(move))

         print("before get agent")
         #agent_pos = self.get_agent_pos(location)
         agent = self.get_agent(location)
         print("after get agent")
         #send agent_pos into do_action
         #print(agent)

         print("The selected move for the agent with id", str(
             agent.unique_id), "and position", agent.pos, "is", str(move))

         #print("About to do action")
         agent.do_action(action, pos, location, agent)
         #self.do_action(action,cell,location)


   def get_agent_pos(self,location):
      return self.get_agent(location).pos

   def get_agent(self, location):
      print("Getting agent now")
      agent = self

      print("self:",str(self),location)

      if location == 0 and not agent.previous == None:
         while not agent.previous == None:
            print('going to previous')
            agent = agent.previous

      elif location == 1 and not agent.next == None:
         while not agent.next == None:
            print('going to next')
            agent = agent.next

      #print('Returning agent')
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
            #self.update_facing(pos)
            self.model.grid.move_agent(self, pos)
            return True

   def eat(self, food_agent):
      self.food += food_agent.consume()
      print("Food for", str(self.unique_id), "is",str(self.food))




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

   def step(self):
      pass

   def get_pos(self):
      return self.pos

   def is_end(self):
      return self.end

class Food(ExtendedAgent):
   def __init__(self, unique_id: int, model: Model, pos=None):
      super().__init__(unique_id, model)
      self.amount = 2
      self.kill_flag = False

   def die(self):
      if self.kill_flag == False:
         self.kill_flag = True
         self.model.kill_agents.append(self)
      #self.model.grid.remove_agent(self)
      #self.model.schedule.remove(self)
      #del self
      return

   def step(self):
      if self.amount <= 0:
         self.die()

   def consume(self):
      print("Running consume")
      amt = 0
      amount = abs(2)
      if self.kill_flag == False:
         if self.amount - amount <= 0:
            amt = self.amount
            self.amount = 0
            #self.die()
         else:
            self.amount -= amount
            amt = amount

      print("Consumed amount is",str(amt))
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
      neighbors = self.neighbors(center = True, moore=False)
      UNDEPLETE = 3
      count = 0

      #problem need to do different neightbors for oxy and

      for cell in neighbors:
         contents = self.model.grid.get_cell_list_contents(cell)
         if isinstance(contents, list) and len(contents) > 1:
            temp_var = False
            for item in contents:
               if isinstance(item, Food): #LarvaAgent)): #LarvaTail)):
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
      self.update()
      return self.traversable

   def step(self):
      self.update()
      pass

   def neighbors(self, pos=None, center = False, moore = True):

      if (pos == None):
         pos = self.pos

      
      neighbors = self.model.grid.get_neighborhood(
          pos,
          moore=True,
          include_center=False)
      return neighbors

   def neighbor_contents(self, pos=None):
      neighbors = self.neighbors(pos)
      return self.model.grid.get_cell_list_contents(neighbors)

class LarvaModel(Model):

   def __init__(self, N, width, height):
      FOOD_HEIGHT = height - (LARVA_SIZE+1)

      print("init model")
      super().__init__()
      self.num_agents = N
      self.schedule = RandomActivation(self)
      self.grid = MultiGrid(width, height, False)
      self.kill_agents = list()
      self.create_agents(width, height)

   def create_agents(self, width, height):
      FOOD_HEIGHT = height - (LARVA_SIZE+1)
      #create agents
      for i in range(self.num_agents):
         agent = LarvaAgent(i, self, "test")
         self.schedule.add(agent)

         #2(i-1)
         x = 2*i
         y = math.floor(FOOD_HEIGHT)
         self.grid.place_agent(agent, (x, y))
         i = 0
         while (not agent.next == None):
            i += 1
            agent = agent.next
            #self.schedule.add(agent)
            self.grid.place_agent(agent, (x, y+i))

      for i in range(height):
         for j in range(width):
            id = str(1000) + str(j) + str(100) + str(i) + str(i*j)
            depleted = True
            traversable = False
            if i >= math.floor(height/2) + 1:
               depleted = False
               traversable = False
            else:
               depleted = True
               traversable = True
            agent = Oxygen(id, self, depleted, traversable)
            self.schedule.add(agent)
            self.grid.place_agent(agent, (j, i))

      for i in range(math.floor(FOOD_HEIGHT)):
          for j in range(width):
             agent = Food(str(2000)+str(j)+str(3)+str(i)+str(3*i+10+j), self)
             self.schedule.add(agent)
             self.grid.place_agent(agent, (j, i))

   def step(self):
      '''Advance the model by one step.'''
      self.schedule.step()
      for x in self.kill_agents:
         print(self.kill_agents)
         print(x)
         print(x.pos)
         self.kill_agents.remove(x)
         self.grid.remove_agent(x)
         self.schedule.remove(x)
