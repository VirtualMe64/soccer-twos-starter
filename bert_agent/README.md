# CEIA Baseline Agent

**Agent name:** Bert

**Author (s):** Samuel Taubman (staubman3@gatech.edu), Aishani Chakraborty (achakraborty75@gatech.edu), Dennis Anthony (danthony33@gatech.edu)

## Description

An agent trained with PPO via multi-agent self-play using Ray RLLib. Additional rewards were added for ball location and proximity to the ball to improve the sparse reward problem. Model size is quite small (single hidden layer of size 128) due to observed massive improvements in convergence time and performance.
