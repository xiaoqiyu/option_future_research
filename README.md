# option_future_research
This repo will focus on hf factor research, clf/reg model research, and backtester for T0 strategy.
 Structures:
    - conf: conf file 
    - global: global define across the projects
    - cache: some historical tick mkt data for testing purpose
    - research: factor and model research, mainly focus on tick model for short-terms prediction,e.g. 5s,10s
    - results: model evaluation results and backtesting results for strategy
    - strategy: strategies are kept here. To create a new strategy, you need to define a new signal class which inherit 
                from base class Signal