from gym_minigrid.minigrid import *
from gym_minigrid.register import register


class Room:
    def __init__(self,
                 top,
                 size,
                 entryDoorPos,
                 exitDoorPos
                 ):
        self.top = top
        self.size = size
        self.entryDoorPos = entryDoorPos
        self.exitDoorPos = exitDoorPos


class MultiAgentEnv(MiniGridEnv):
    """
    Environment with multiple rooms (subgoals)
    """

    def __init__(self,
                 minNumRooms,
                 maxNumRooms,
                 maxRoomSize=10,
                 agents_type=None,
                 view_size=7
                 ):
        assert minNumRooms > 0
        assert maxNumRooms >= minNumRooms
        assert maxRoomSize >= 4

        self.minNumRooms = minNumRooms
        self.maxNumRooms = maxNumRooms
        self.maxRoomSize = maxRoomSize

        self.rooms = []
        self.door_num = 0

        if agents_type is None:
            self.agents_type = ['door_opening', 'goal_reaching']
        else:
            self.agents_type = agents_type
        # self.agents_type = ['door_opening']
        # self.agents_type = ['goal_reaching']

        agents = []
        if 'door_opening' in self.agents_type:
            agents.append(Agent(0, view_size=view_size, agent_type='door_opening'))
        if 'goal_reaching' in self.agents_type:
            agents.append(Agent(1, view_size=view_size, agent_type='goal_reaching'))

        max_steps = 500 if len(self.agents_type) == 2 else self.maxNumRooms * 20
        super(MultiAgentEnv, self).__init__(
            grid_size=25,
            max_steps=max_steps,
            agents=agents,
            agent_view_size=view_size
        )

    def _gen_grid(self, width, height):
        self.door_num = 0
        roomList = []

        # Choose a random number of rooms to generate
        numRooms = self._rand_int(self.minNumRooms, self.maxNumRooms + 1)

        while len(roomList) < numRooms:
            curRoomList = []

            entryDoorPos = (
                self._rand_int(0, width - 2),
                self._rand_int(0, width - 2)
            )

            # Recursively place the rooms
            self._placeRoom(
                numRooms,
                roomList=curRoomList,
                minSz=4,
                maxSz=self.maxRoomSize,
                entryDoorWall=2,
                entryDoorPos=entryDoorPos
            )

            if len(curRoomList) > len(roomList):
                roomList = curRoomList

        self._shiftRoom(roomList)

        # Store the list of rooms in this environment
        assert len(roomList) > 0
        self.rooms = roomList

        # Create the grid
        self.grid = Grid(width, height)
        wall = Wall()

        prevDoorColor = None

        # For each room
        for idx, room in enumerate(roomList):

            topX, topY = room.top
            sizeX, sizeY = room.size

            # Draw the top and bottom walls
            for i in range(0, sizeX):
                self.grid.set(topX + i, topY, wall)
                self.grid.set(topX + i, topY + sizeY - 1, wall)

            # Draw the left and right walls
            for j in range(0, sizeY):
                self.grid.set(topX, topY + j, wall)
                self.grid.set(topX + sizeX - 1, topY + j, wall)

            # If this isn't the first room, place the entry door
            if idx > 0:
                if 'door_opening' in self.agents_type:
                    # Pick a door color different from the previous one
                    # doorColors = set(COLOR_NAMES)
                    # if prevDoorColor:
                    #     doorColors.remove(prevDoorColor)
                    # # Note: the use of sorting here guarantees determinism,
                    # # This is needed because Python's set is not deterministic
                    # doorColor = self._rand_elem(sorted(doorColors))
                    doorColor = IDX_TO_COLOR[0]

                    entryDoor = ColorDoor(doorColor)
                    self.grid.set(*room.entryDoorPos, entryDoor)
                    prevDoorColor = doorColor

                    prevRoom = roomList[idx - 1]
                    prevRoom.exitDoorPos = room.entryDoorPos
                    self.door_num += 1
                else:
                    self.grid.set(*room.entryDoorPos, None)

        # Randomize the starting agent position and direction
        for a in self.agents:
            self.place_agent(a, roomList[0].top, roomList[0].size)

        # Place the final goal in the last room
        if 'goal_reaching' in self.agents_type:
            self.goal_pos = self.place_goal(self.goal, roomList[-1].top, roomList[-1].size)

        self.mission = 'traverse the rooms to get to the goal'

    def _placeRoom(
            self,
            numLeft,
            roomList,
            minSz,
            maxSz,
            entryDoorWall,
            entryDoorPos
    ):
        # Choose the room size randomly
        sizeX = self._rand_int(minSz, maxSz + 1)
        sizeY = self._rand_int(minSz, maxSz + 1)

        # The first room will be at the door position
        if len(roomList) == 0:
            topX, topY = entryDoorPos
        # Entry on the right
        elif entryDoorWall == 0:
            topX = entryDoorPos[0] - sizeX + 1
            y = entryDoorPos[1]
            topY = self._rand_int(y - sizeY + 2, y)
        # Entry wall on the south
        elif entryDoorWall == 1:
            x = entryDoorPos[0]
            topX = self._rand_int(x - sizeX + 2, x)
            topY = entryDoorPos[1] - sizeY + 1
        # Entry wall on the left
        elif entryDoorWall == 2:
            topX = entryDoorPos[0]
            y = entryDoorPos[1]
            topY = self._rand_int(y - sizeY + 2, y)
        # Entry wall on the top
        elif entryDoorWall == 3:
            x = entryDoorPos[0]
            topX = self._rand_int(x - sizeX + 2, x)
            topY = entryDoorPos[1]
        else:
            assert False, entryDoorWall

        # If the room is out of the grid, can't place a room here
        if topX < 0 or topY < 0:
            return False
        if topX + sizeX > self.width or topY + sizeY >= self.height:
            return False

        # If the room intersects with previous rooms, can't place it here
        for room in roomList[:-1]:
            nonOverlap = \
                topX + sizeX < room.top[0] or \
                room.top[0] + room.size[0] <= topX or \
                topY + sizeY < room.top[1] or \
                room.top[1] + room.size[1] <= topY

            if not nonOverlap:
                return False

        # Add this room to the list
        roomList.append(Room(
            (topX, topY),
            (sizeX, sizeY),
            entryDoorPos,
            None
        ))

        # If this was the last room, stop
        if numLeft == 1:
            return True

        # Try placing the next room
        for i in range(0, 8):

            # Pick which wall to place the out door on
            wallSet = set((0, 1, 2, 3))
            wallSet.remove(entryDoorWall)
            exitDoorWall = self._rand_elem(sorted(wallSet))
            nextEntryWall = (exitDoorWall + 2) % 4

            # Pick the exit door position
            # Exit on right wall
            if exitDoorWall == 0:
                exitDoorPos = (
                    topX + sizeX - 1,
                    topY + self._rand_int(1, sizeY - 1)
                )
            # Exit on south wall
            elif exitDoorWall == 1:
                exitDoorPos = (
                    topX + self._rand_int(1, sizeX - 1),
                    topY + sizeY - 1
                )
            # Exit on left wall
            elif exitDoorWall == 2:
                exitDoorPos = (
                    topX,
                    topY + self._rand_int(1, sizeY - 1)
                )
            # Exit on north wall
            elif exitDoorWall == 3:
                exitDoorPos = (
                    topX + self._rand_int(1, sizeX - 1),
                    topY
                )
            else:
                assert False

            # Recursively create the other rooms
            success = self._placeRoom(
                numLeft - 1,
                roomList=roomList,
                minSz=minSz,
                maxSz=maxSz,
                entryDoorWall=nextEntryWall,
                entryDoorPos=exitDoorPos
            )

            if success:
                break

        return True

    def _shiftRoom(self, roomList):
        shiftx = 50
        shifty = 50
        for room in roomList:
            shiftx = room.top[0] if room.top[0] < shiftx else shiftx
            shifty = room.top[1] if room.top[1] < shifty else shifty
        for room in roomList:
            room.top = (room.top[0] - shiftx, room.top[1] - shifty)
            room.entryDoorPos = (room.entryDoorPos[0] - shiftx, room.entryDoorPos[1] - shifty)
        return True


class MultiAgentEnvN2S4(MultiAgentEnv):
    def __init__(self):
        super().__init__(
            minNumRooms=2,
            maxNumRooms=2,
            maxRoomSize=4,
            agents_type=None
        )


class MultiAgentEnvN2S4R(MultiAgentEnv):
    def __init__(self):
        super().__init__(
            minNumRooms=2,
            maxNumRooms=2,
            maxRoomSize=4,
            agents_type=['door_opening']
        )


class MultiAgentEnvN2S4G(MultiAgentEnv):
    def __init__(self):
        super().__init__(
            minNumRooms=2,
            maxNumRooms=2,
            maxRoomSize=4,
            agents_type=['goal_reaching']
        )


class MultiAgentEnvN4S5(MultiAgentEnv):
    def __init__(self):
        super().__init__(
            minNumRooms=4,
            maxNumRooms=4,
            maxRoomSize=5,
            agents_type=None
        )


class MultiAgentEnvN4S5R(MultiAgentEnv):
    def __init__(self):
        super().__init__(
            minNumRooms=4,
            maxNumRooms=4,
            maxRoomSize=5,
            agents_type=['door_opening']
        )


class MultiAgentEnvN4S5G(MultiAgentEnv):
    def __init__(self):
        super().__init__(
            minNumRooms=4,
            maxNumRooms=4,
            maxRoomSize=5,
            agents_type=['goal_reaching']
        )


class MultiAgentEnvN6(MultiAgentEnv):
    def __init__(self):
        super().__init__(
            minNumRooms=6,
            maxNumRooms=6,
            agents_type=None
        )


register(
    id='MiniGrid-MultiAgent-N2-S4-v0',
    entry_point='gym_minigrid.envs:MultiAgentEnvN2S4'
)

register(
    id='MiniGrid-MultiAgent-N2-S4-A1R-v0',
    entry_point='gym_minigrid.envs:MultiAgentEnvN2S4R'
)

register(
    id='MiniGrid-MultiAgent-N2-S4-A1G-v0',
    entry_point='gym_minigrid.envs:MultiAgentEnvN2S4G'
)

register(
    id='MiniGrid-MultiAgent-N4-S5-v0',
    entry_point='gym_minigrid.envs:MultiAgentEnvN4S5'
)

register(
    id='MiniGrid-MultiAgent-N4-S5-A1R-v0',
    entry_point='gym_minigrid.envs:MultiAgentEnvN4S5R'
)

register(
    id='MiniGrid-MultiAgent-N4-S5-A1G-v0',
    entry_point='gym_minigrid.envs:MultiAgentEnvN4S5G'
)

register(
    id='MiniGrid-MultiAgent-N6-v0',
    entry_point='gym_minigrid.envs:MultiAgentEnvN6'
)
