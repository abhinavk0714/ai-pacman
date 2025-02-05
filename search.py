# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import searchAgents
import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    # initialize stack
    stack = util.Stack()

    # push start state and empty instructions
    stack.push((problem.getStartState(), []))

    # set of visited states
    visited = set()

    while not stack.isEmpty():
        currentState, actions = stack.pop()
        # if at goal, return
        if problem.isGoalState(currentState):
            return actions
        # keep searching unvisited states
        if currentState not in visited:
            visited.add(currentState)
            for successor, action, stepCost in problem.getSuccessors(currentState):
                if successor not in visited:
                    stack.push((successor, actions + [action]))

    return []


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    # initialize queue
    queue = util.Queue()

    # push start state and empty instructions
    queue.push((problem.getStartState(), []))

    # set of visited states
    visited = set()

    while not queue.isEmpty():
        currentState, actions = queue.pop()
        # if at goal, return the actions
        if problem.isGoalState(currentState):
            return actions
        # keep searching unvisited states
        if currentState not in visited:
            visited.add(currentState)
            for successor, action, stepCost in problem.getSuccessors(currentState):
                if successor not in visited:
                    queue.push((successor, actions + [action]))

    return []


def iterativeDeepeningSearch(problem):
    """Search the tree iteratively for goal nodes."""
    depth = 0  # initialize depth limit

    while True:
        # initialize everything
        stack = util.Stack()
        seen = set()
        visited = set()

        # push all successors of the start state to the stack
        for successor, action, cost in problem.getSuccessors(problem.getStartState()):
            stack.push((successor, [action], cost, 1))
            seen.add(successor)

        # mark the start state as seen and visited
        startState = problem.getStartState()
        seen.add(startState)
        visited.add(startState)

        # depth-limited depth-first search
        while not stack.isEmpty():
            currentState, actions, totalCost, currentDepth = stack.pop()

            if currentDepth <= depth:  # only process nodes within the depth limit
                if currentState in visited:
                    continue  # skip nodes already fully explored
                
                if problem.isGoalState(currentState):
                    return actions  # return the solution path if goal is found

                visited.add(currentState)  # mark the current state as visited

                # get successors of the current state
                for successor, action, cost in problem.getSuccessors(currentState):
                    if (successor not in visited and successor not in seen and currentDepth + 1 <= depth):
                        newActions = actions + [action]  # extend the action path
                        stack.push((successor, newActions, totalCost + cost, currentDepth + 1))  # add successor to stack
                        seen.add(successor)  # mark successor as seen
        
        depth += 1  # increment depth limit


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    # initialize priority queue
    priorityQueue = util.PriorityQueue()

    # push start state with cost 0
    priorityQueue.push((problem.getStartState(), [], 0), 0)

    # dict to store visited nodes w the least cost
    visited = {}

    while not priorityQueue.isEmpty():
        currentState, actions, totalCost = priorityQueue.pop()
        # if at goal, return the actions
        if problem.isGoalState(currentState):
            return actions
        # if the state hasn't been visited or there is a cheaper path found
        if currentState not in visited or totalCost < visited[currentState]:
            visited[currentState] = totalCost
            for successor, action, stepCost in problem.getSuccessors(currentState):
                newCost = totalCost + stepCost
                priorityQueue.push((successor, actions + [action], newCost), newCost)

    return []


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    # initialize priority queue
    priorityQueue = util.PriorityQueue()

    # push start state with cost 0
    startState = problem.getStartState()
    priorityQueue.push((startState, [], 0), heuristic(startState, problem))

    # dict to store visited nodes with the least cost
    visited = {}

    while not priorityQueue.isEmpty():
        currentState, actions, totalCost = priorityQueue.pop()
        # if at goal, return the actions
        if problem.isGoalState(currentState):
            return actions
        # if the state hasn't been visited or there is a cheaper path found
        if currentState not in visited or totalCost < visited[currentState]:
            visited[currentState] = totalCost
            for successor, action, stepCost in problem.getSuccessors(currentState):
                newCost = totalCost + stepCost
                # find the heuristic cost and total priority
                heuristicCost = heuristic(successor, problem)
                totalPriority = newCost + heuristicCost
                # push onto the queue with the total priority
                priorityQueue.push((successor, actions + [action], newCost), totalPriority)

    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
iddfs = iterativeDeepeningSearch