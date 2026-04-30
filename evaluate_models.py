import soccer_twos
from soccer_twos.evaluate import collect_episodes, load_agent

env = soccer_twos.make()

def evaluate_pair(agent_1, agent_1_name, agent_2, agent_2_name, n_episodes=1000):
    data = collect_episodes(env, agent_1, agent_2, n_episodes=n_episodes)
    winners = []
    for d in data:
        if d['agent_1_reward'] > d['agent_2_reward']:
            winners.append(agent_1_name)
        else:
            winners.append(agent_2_name)

    return winners

agents = [["bert_agent", "Bert Agent"], ["sammy_agent", "Reward Shaping"], ["ceia_baseline_agent", "Baseline"], ["example_player_agent", "Random Agent"]]
for a in agents:
    a[0] = load_agent(a[0])

output = []
for i in range(len(agents)):
    for j in range(i + 1, len(agents)):
        m1 = agents[i]
        m2 = agents[j]
        winners = evaluate_pair(m1[0], m1[1], m2[0], m2[1])
        output.append((m1[1], m2[1], winners.count(m1[1]), winners.count(m2[1])))

for o in output:
    print(f"{o[0]} vs {o[1]}: {o[2]} wins for {o[0]}, {o[3]} wins for {o[1]}")