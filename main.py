"""Mini Campus Delivery Robot: A*, Bayes, and one-step MDP. Python 3.8+."""
import argparse
import heapq

SIZE = 5
WALLS = {(1, 1), (2, 1)}
START, GOAL = (0, 0), (4, 4)
STATES = ('clear', 'blocked')
# T[action][state] is a probability distribution over next corridor states.
TRANSITIONS = {
    'Move': {'clear': {'clear': .9, 'blocked': .1},
             'blocked': {'clear': .2, 'blocked': .8}},
    'Wait': {'clear': {'clear': .95, 'blocked': .05},
             'blocked': {'clear': .6, 'blocked': .4}},
}
# Immediate reward depends on the CURRENT state, before any transition.
REWARDS = {'Move': {'clear': 10., 'blocked': -20.},
           'Wait': {'clear': -2., 'blocked': -1.}}


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(start, goal, walls, size=5):
    """Four-direction unit-cost A*. Return optimal path and expansion trace."""
    def valid(p):
        return 0 <= p[0] < size and 0 <= p[1] < size and p not in walls
    if not valid(start) or not valid(goal):
        return [], []
    frontier = [(manhattan(start, goal), 0, start)]
    costs, parent, trace = {start: 0}, {}, []
    while frontier:
        f, g, current = heapq.heappop(frontier)
        if g != costs[current]:
            continue  # Ignore outdated queue entries.
        trace.append((current, g, manhattan(current, goal), f))
        if current == goal:
            path = [goal]
            while path[-1] != start:
                path.append(parent[path[-1]])
            return list(reversed(path)), trace
        for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            nxt = (current[0] + dr, current[1] + dc)
            if valid(nxt) and g + 1 < costs.get(nxt, float('inf')):
                costs[nxt], parent[nxt] = g + 1, current
                heapq.heappush(frontier, (g + 1 + manhattan(nxt, goal), g + 1, nxt))
    return [], trace


def bayes(prior, alert, sensitivity=.85, false_positive=.15):
    """P(blocked | observation); sensor rates are explicit toy assumptions."""
    for value in (prior, sensitivity, false_positive):
        if not 0 <= value <= 1:
            raise ValueError('Probabilities must lie between 0 and 1.')
    likelihood_blocked = sensitivity if alert else 1 - sensitivity
    likelihood_clear = false_positive if alert else 1 - false_positive
    numerator = prior * likelihood_blocked
    evidence = numerator + (1 - prior) * likelihood_clear
    if evidence == 0:
        raise ValueError('Observation has zero probability under this model.')
    return numerator / evidence, likelihood_blocked, likelihood_clear, evidence


def mdp_decision(p_blocked, gamma=.9, future_values=None):
    """One Bellman backup, then belief-weighted action selection.

    Q(s,a) = R(s,a) + gamma * sum T(s,a,s') V0(s').
    V0=0 by default: the assignment permits a one-step reward comparison.
    The uncertain current state is summarized by the Bayesian belief.
    """
    if not 0 <= p_blocked <= 1 or not 0 <= gamma <= 1:
        raise ValueError('Invalid probability or discount factor.')
    values = dict.fromkeys(STATES, 0.) if future_values is None else future_values
    belief = {'clear': 1 - p_blocked, 'blocked': p_blocked}
    q, scores = {}, {}
    for action in REWARDS:
        q[action] = {}
        for state in STATES:
            distribution = TRANSITIONS[action][state]
            continuation = sum(prob * values[nxt] for nxt, prob in distribution.items())
            q[action][state] = REWARDS[action][state] + gamma * continuation
        scores[action] = sum(belief[s] * q[action][s] for s in STATES)
    # Conservative tie-break: Wait.
    action = 'Move' if scores['Move'] > scores['Wait'] else 'Wait'
    return action, scores, q


def draw_map(path, position=START):
    print('    ' + ' '.join(str(c) for c in range(SIZE)))
    for r in range(SIZE):
        cells = []
        for c in range(SIZE):
            p = (r, c)
            cells.append('R' if p == position else 'G' if p == GOAL else
                         '#' if p in WALLS else '*' if p in path else '.')
        print(f'{r}   ' + ' '.join(cells))
    print('R=robot G=goal #=known wall *=planned route; coordinates=(row,column)')


def demo(alert, prior=.3, trace=False):
    print('\n' + '=' * 64)
    print('MINI CAMPUS DELIVERY ROBOT | Sensor: ' + ('ALERT' if alert else 'CLEAR'))
    print('Toy simulation: all probabilities, rewards and walls are predefined.')
    path, steps = astar(START, GOAL, WALLS, SIZE)
    print('\n1. A* SEARCH: start=(0,0), goal=(4,4), two known walls')
    draw_map(path)
    if not path:
        print('No route exists. Stop safely; no movement decision is executed.')
        return
    print('Path:', ' -> '.join(map(str, path)))
    print(f'Path length: {len(path)-1} moves; expanded nodes: {len(steps)}')
    print('Calculation: f(n)=g(n)+h(n); h=Manhattan distance.')
    print('A* finds a shortest route around known walls.')
    if trace:
        print('Expanded node | g | h | f')
        for node, g, h, f in steps:
            print(f'{node!s:13} | {g} | {h} | {f}')
    p, lb, lc, evidence = bayes(prior, alert)
    print(f'\n2. BAYES: sensor observes next planned cell {path[1]}')
    print(f'Prior P(blocked)={prior:.3f}; sensitivity=.85; false positive=.15')
    print(f'P(observation|blocked)={lb:.3f}; P(observation|clear)={lc:.3f}')
    print(f'P(blocked|observation)=({prior:.3f}*{lb:.3f})/{evidence:.3f}={p:.6f}')
    print('Bayes updates uncertainty about an unknown temporary obstacle.')
    action, scores, q = mdp_decision(p)
    print('\n3. ONE-STEP MDP: states=clear/blocked; actions=Move/Wait')
    print('Q(s,a)=R(s,a)+0.9*sum(T(s,a,s\')*V0(s\')); V0=0')
    for a in REWARDS:
        for s in STATES:
            print(f'  {a}, {s}: T={TRANSITIONS[a][s]}, R={REWARDS[a][s]:g}, Q={q[a][s]:g}')
        print(f'  Expected {a}: (1-{p:.6f})*{q[a]["clear"]:g}'
              f' + {p:.6f}*{q[a]["blocked"]:g} = {scores[a]:.6f}')
    print('MDP compares actions using the Bayesian state probabilities.')
    print(f'RECOMMENDATION: {action.upper()}')
    position = path[1] if action == 'Move' else START
    print(f'Executed in simulation: {START} -> {position}')
    print('Movement is a commanded step, not proof that a hidden cell is free.')
    return {'path': path, 'posterior': p, 'action': action, 'scores': scores}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', choices=('both', 'alert', 'clear'), default='both')
    parser.add_argument('--prior', type=float, default=.3)
    parser.add_argument('--trace', action='store_true', help='Print every A* expansion')
    args = parser.parse_args()
    if not 0 <= args.prior <= 1:
        parser.error('--prior must be between 0 and 1')
    for alert in ([True, False] if args.scenario == 'both' else [args.scenario == 'alert']):
        demo(alert, args.prior, args.trace)


if __name__ == '__main__':
    main()
