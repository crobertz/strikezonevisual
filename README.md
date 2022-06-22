# strikezonevisual
Tools for visualizing MLB strike zone using StatCast data.

StatCast provides pitch data for every single pitch throw in a MLB game, including its location and whether it was called strike or ball.
Using this data one can easily plot pitches thrown in a game and filter by various criteria such as the player or the umpire.

## Learning the strike zone

Since StatCast provides data for each pitch and whether it was called strike or ball, it naturally sets itself up to a binary classification problem of learning the strike zone.

### Umpire strike zone

In this project we group each pitch by the home plate umpire and use a simple MLP binary classification to learn his strike zone and output the visual.
Since StatCast does not provide home plate umpire data, we need to obtain that data elsewhere and merge with the StatCast data.
Here home plate umpire information for the 2021 season was taken from Retrosheet and combined with all pitch data from 2021.

Once we have a CSV with StatCast pitch data along with the plate umpire information it is a simple matter of training a MLP to learn the strike zone and output the decision boundary overlaid with the "true" strike zone. We can further filter the data before visualization for example by handedness, matchup, etc.

### Sample Output
Below is a plot of all balls and strikes called by Angel Hernandez over the 2021 MLB season.

![Sample output](https://github.com/crobertz/strikezonevisual/blob/main/out/Angel_Hernandez.png)

An MLP is trained on the strikes and balls data and the decision boundary plotted along with an overlay for the true strike zone.
The horizontal width of the true zone is the width of home plate which is 17 inches.
The top and bottom vertical positions of the true zone are given by taking the average of all `sz_top` and `sz_bot` values in the StatCast csv file respectively.
