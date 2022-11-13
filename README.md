...
export FLASK_APP-api/app.py ; flask run
...

...
docker build -t exp:v1 -f docker/Dockerfile 
docker run -it exp:v1
...













The comparision table with mean and standard deviation is:
First_column    |  svm | decision_tree
|--------------|-------|-------------|
|           1 |0.983240 |      0.837989
|           2 |0.994413      | 0.860335
|           3 |0.988827      | 0.871508
|           4 |0.988827      | 0.899441
|           5 |0.988827      | 0.810056
|        mean |0.988827      | 0.855866
|         std |0.003950      | 0.033844