# InsuranceCustomersSegmentation_MachineLearning
## Objective

Assuming the role of a consultant, the project consisted on developing a Customer Segmentation in such a way that it will be possible for the Marketing Department of an insurance company to understand all the different Customers’ Profiles better.

Employing the CRISP-DM process model, you are expected to define, describe and explain the clusters you chose. Invest time in reasoning how you want to do your clustering, possible approaches, and advantages or disadvantages of different decisions. Simultaneous, you should express the marketing approach you recommend for each cluster.

This project aims to develop a customer segmentation of an insurance company’s clients, in order for the marketing department to understand better its customers and hence apply the most suitable strategies. The considered models will be done through cluster analysis to determine similarities and differences between different clients’ profiles.

## Dataset
By checking some initial metrics of the raw data, it is possible to notice the presence of missing values, since the frequency of each variable is not consistent and varies. The maximum “count” is 10 296, which goes in line with the Customer ID. Thus, it is supposed to have data of 10 296 clients of the insurance company.

The only unique value presented is in the Education Degree feature, which doesn’t go in line with the other customers’ classifications. The most frequent degree is Bachelor or Masters, having a frequency of 4 799, which is equivalent to 46,70% of the customers presenting this educational level.

Concerning the year of the first policy in the insurance company, it ranges from 1974 and 53784, which is most certainly due to a typo when someone registered this client. The average first policy year is 1991 with a median of 1986.

Regarding the clients’ age, there’s also a typo when registering a client, since the minimum birth year is 1028 and it is impossible for someone having 988 years (considering the year 2016). In terms of frequency, the average client has 48 years old (1968) and the youngest customer has 15 years (2001), which most certainly is not related to a car insurance policy.

By checking the monthly salary, the clients’ average is 2 507€, with a significant standard deviation of 1 157€. The lower bound is 333€, with the highest salary totalling 55 125€. The median presents a value near the average, being 2 502€.
The most frequent area of living is code number 3 and the number of dependents is usually 1 child.

By looking at costs, customers provide, on average, a value of 177,89 (Customer Monetary Value), with the costly customer being way superior to the most profitable one (-165 680 vs. 11 876). In the past 2 years of the data, the amount paid by the insurance is below the premiums received, averaging a Claim Rate of 0.74.

Finally, considering the different types of insurance policies, the insurance company receives, on average, more funds from the Motor Line of Business, followed by the Household line. However, the highest premiums received are also related to the Household but also from the Health Business Line.

## Data Understanding
The data contains 14 columns with information on education, birth year, salary, family and information about the amount paid by types of insurance products.

The only categorical variable of the dataset is the Education Degree which is divided into 4 categories: BSc/MSc (bachelor or masters), High School, Basic and PhD. Most of the insurance clients have a master or a bachelor degree, while PhD students are less frequent.

Almost all features present missing values except “CustMonVal”, “ClaimsRate” and “PremHousehold”.

Plotting the considered variables in a histogram, the previously obtained statistics for this dataset can be verified.

Starting with the variable “FirstPolYear”, almost the entirety of the observations is comprehended between 1974 and 1992.

Moving forward to the “BirthYear”, the histogram shows that the range of values for this variable is quite limited as well, being all below 2001.

As for the “MonthSal” variable, the observations for this variable are all concentrated on the lower part of the horizontal axis, being almost all below 3.300 except for some outliers.

The “GeoLivArea” histogram is composed by four different bars, each representing one of the four categories that compose this variable. Through this, one is able to conclude that the majority of the observations have as attribute the category 4 of this variable and the category 2 is the one that is observed in a smaller number of observations in the dataset.

“Children” is a binary variable that assumes the values 0 or 1. The histogram shows that most observations of this dataset have, at least, 1 child.

When analysing the variables “CustMonVal” through the histogram, it is clear that the majority of observations are concentrated in a range of values not far from each other. This same conclusion also applies for the variables “ClaimsRate”, “PremMot”, “PremHousehold”, “PremHealth” and “PremWork”, despite this last variable containing a smaller number of observations with higher values, closer to 500.

Finally, to what concerns variable “PremLife”, the histogram clearly presents a distribution that is skewed to the right.

The above plotted boxplots are useful to conclude that some of the considered variables present some values that are clearly distant from the others. “FirstPolYear” presents a single observation that differs from the others and was, therefore, considered and treated as an outlier. Similarly, the “BirthYear” variable boxplot shows a single point far from the majority of the other values of this variable. This was also considered an outlier of the dataset.

Adding to these two described situations, the boxplots from the variables “MonthSal”, “CustMonVal” and “ClaimsRate” also present some dispersion. In other words, there are some observations that are quite different when considering the universe of observations of that variable. Although, considering what is the meaning of these variables and what these represent, they were not considered as outliers.

Pearson Correlation measures the strength of the linear relationship between two variables. A correlation heatmap was plotted to graphically represent the correlation between all the different variables.

In the graph above, it can be highlighted that the Claim Rate and the Customer Monetary Value have an almost perfect negative correlation, as the claim rate increases the monetary value of the costumer decreases. When the claim rate increases, the insurance company will suffer more costs therefore the costumers monetary value will decrease since its annual profit also decreases. On the other hand, the Monthly Salary and the Birth year also have a strong correlation, when the salary increases the birth year decreases, meaning that as a costumers’ age increases its monthly income increases too.

## Data Preparation
The first step in preparing the data was to remove the Costumer ID variable since it was not considered relevant for the development of customers’ profiles. For the BirthYear and FirstPolYear variables, two values were removed from the sample since they were considered typo errors.

The missing values were treated for almost all of the variables. In the variables FirstPolYear and Birth Year the median was choosen to fill the missing values. For the variables Monthly Salary, PremMotor, PremHealth, PremLife and PremWork the mean was used to replace the missing values, since the average values would not add value to the data set but keep it balanced.

The missing values in the Children variable represent a no response, so it was considered that those costumers did not have children. In the living area there was only one missing value, so it was decided to drop that value has it would not affect the data set. The missing values for the Education degree were replaced for the most frequent category, in this case Bsc and Msc, and additionally the categories for this variable were transformed into numerical ones for easier analysis.

Lastly, the Monthly Salary was divided in intervals of 1000 euros until 5000 euros, in order to better represent this variable and provide further analysis for costumer profiling.

### Analyze principal components
The PCA technique consists of reducing the dimensionality of the data to increase interpretability while minimizing information loss. It does so by creating new and uncorrelated variables that can maximize variance. It was applied the PCA method to visualize all the components and the power of each other and how much of them can explain the data.

It could be inferred that the number of components that has a variance explained make the highest point between 7 and 9 dimensions when compared to 17 of all datasets.

## K-Means Clustering (7 Components)
The K-Means clustering is an unsupervised machine learning algorithm that groups data points through an iterative procedure. Is a partitional clustering methodology since a data point can only be assigned to a single cluster, meaning that different clusters can’t possess the same data point.

The algorithm starts by assigning a random number (K) of centroids – the centre of a cluster. Then, it calculates the distance (e.g. euclidean distance) of each data point to the centroids and assigns each datapoint to the nearest centroid. Afterwards, it calculates the average of each centroid’s data points and it substitutes the centroids’ location to the average and so on. The algorithm repeats this step until obtains some kind of convergence between the data.

In order to define the number of clusters (K) to implement in the model, it was implemented the elbow and silhouette method, as we’ve seen in class. The defined range of K in K-means was 1 to 20, as suggested by the professor.

From the graphical visualization of the elbow method, it is possible to observe that the optimal number of K is 7, meaning that the interception is the point where the distortion (Within-Cluster-Sum of Squared Errors) starts to gradually decrease.

The Silhouette method scores the similarity between the datapoints and its clusters when compared to other clusters. By applying this method, the number of clusters that outputs a better score is a K = 6. Although that is verified graphically when compared to K = 7, it is possible to observe that a K = 6 produces a somewhat disproportionate clusters.

When plotting the cluster frequency with K = 6 it is possible to observe that clusters 4 and 5 have the lowest frequency. When plotting the frequency for a K = 7 there’s also low presence of clusters 0 and 6 in comparison to the others. In terms of magnitude, cluster 4 is the lowest for K = 6 and for K = 7 are again clusters 0 and 6.

By comparing the relationship between frequency and magnitude, it is possible to observe that a K = 6 produces more deviant results. However, when visualizing the intercluster distances, a K = 6 produces overall closest results, although K = 7 outputs closer results between pairs.

Given these results, 7 clusters were applied to the dataset since the data seemed, globally, more balanced than using 6 clusters.

## Cluster Insights
### Cluster 0: Young adults with low purchasing power

 - Most have a monthly income of 1000 euros maximum,
 - The less educated persons (most with only basic education)
 - Most have, at least, 1 child
 - Young adults, most being around 22 years old
 - Present a relatively low percentage of claims rate
 - Lowest Motor premiums and highest house premiums
 - Most profitable segment
 
**Likely to prefer more basic insurance plans**

### Cluster 1: Senior people with no children and high purchasing power

 - All customers have a monthly income between 3000 and 4000 euros
 - Educated people (Bachelor/ Masters degree)
 - No children
 - Senior people, on average 69 years
 - Low percentage of claims rate
 - Similar premiums in motor, housing and health

**Likely to prefer more premium insurance plans**

### Cluster 2: Middle-aged customers with medium purchasing power

 - Educated people
 - Monthly Salaries between 2000 and 3000 euros
 - Most have, at least, 1 child
 - Average birth Year of 1967, being characterized by people of around 55 years old
 - The second highest PremMotor (378,79)
 - Low PremLife, PremWork and PremHealth

**Likely to prefer more standard insurance plans**

### Cluster 3: Educated young adults with low purchasing power

 - Young adults, born in 1985, on average
 - Educated people
 - Monthly salary between 1000 and 2000 euros
 - Second highest claims rate
 - Second less profitable

**Likely to prefer more basic insurance plans**

### Cluster 4: Profitable Customers, in their middle age with lowest claim rate.

 - Middle-class salary between 2000 and 3000 euros
 - Educated people
 - Most of the customers have children
 - Middle-aged people with, on average, 48 years
 - Living area – region 4
 - Highest Motor premiums although presenting the lowest claims rate
 - Lowest Health premiums

**Likely to prefer more standard plans**

### Cluster 5: Senior customers with children and high purchasing power

 - All customers have a monthly salary between 3000 and 4000 euros
 - The average birth year is 1956, representing customers around the age of 60 years.
 - Educated people
 - All the customers have at least one child.
 - Presents high motor premiums and the third-highest claims rate
 - Presents similar premiums in Health and Housing

**Likely to prefer more premium/full coverage plans**

### Cluster 6: Oldest customers with no children and high purchasing power

 - Highest purchasing power with a monthly income above 4000 euros
 - Educated people
 - Most have no children
 - Highest health premium
 - Less profitable
 - Highest claims rate
 
**Likely to prefer more premium/full coverage plans**

## Hierarchical clustering
This clustering technique is a very useful segmentation tool. In a simple way, this method consists of, firstly, considering certain points (the points that one wants to cluster into groups), assigning separate clusters to each one of these points. Afterwards, taking into account possible similarities between these points, combining the ones that are most similar together. This process is repeated until only one single cluster is left, building a hierarchy of clusters. This type of clustering can be either agglomerative or divisive.

**Hope you find this project interesting**
