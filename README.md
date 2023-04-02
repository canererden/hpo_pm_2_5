# hpo_pm_2_5
Paper: https://link.springer.com/article/10.1007/s13762-023-04763-6

## Abstract
Since air pollution negatively affects human health and causes serious diseases, accurate air pollution prediction is essential regarding environmental sustainability. Although conventional statistical and machine learning methods have been widely used for air quality forecasting, they have limitations in finding nonlinear relations and modeling sequential data. In recent years, deep learning methods such as long short-term memory, recurrent neural networks, and gated recurrent units have been successfully applied in several research areas, including time-series forecasting. In this study, deep learning algorithm is employed to predict the PM2.5 dataset, including air pollutants (NO, NO2, NOX, O3, PM2.5, SO, and SO2) and meteorological features (wind speed, wind direction, and air temperature) in Istanbul metropolitan. Deep learning algorithms have many hyperparameters such as learning and dropout rate, the number of hidden layers and units in each hidden layer, activation function, loss function, and optimizer that need to be optimized in order to achieve optimal training performance. Therefore, a genetic algorithm-based hyperparameter optimization approach is proposed to find the best parameter combination. The prediction results of deep learning algorithms are compared with default hyperparameters and random search algorithms to confirm the efficacy of the genetic algorithm approach. The proposed method outperforms the other configurations, with the MSE error reduced by 13.38\% and 55.30\% for testing performance, respectively. The experimental results revealed that genetic algorithms are promising and applicable in hyperparameter optimization of deep neural network models, especially in air quality forecasting.

## Reference
```
@article{erden_genetic_2023,
	title = {Genetic algorithm-based hyperparameter optimization of deep learning models for {PM2}.5 time-series prediction},
	volume = {20},
	issn = {1735-2630},
	url = {https://doi.org/10.1007/s13762-023-04763-6},
	doi = {10.1007/s13762-023-04763-6},
	number = {3},
	journal = {International Journal of Environmental Science and Technology},
	author = {Erden, C.},
	month = mar,
	year = {2023},
	pages = {2959--2982},
}
```

