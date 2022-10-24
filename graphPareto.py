
#import shit

def plotGraph(list):

    #list will be a list of x and y values
    #each element in the list will be another sublist of 2 elements with xValues = sublist[0] and
    #yValues = sublist[1]

    #example xValues: fitness_1 = [ind.fitness.values[0] for ind in hof] from titanic_MOGP.py

    #example yValues: fitness_2 = [ind.fitness.values[1] for ind in hof] from titanic_MOGP.py

    #can get pareto front of individuals from hof from evolution loop

    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
    int i = 0
    AUCs = []
    for (subList in list):
        currentColor = colors[i]
        xValues = subList[0]
        yValues = subList[1]
        plt.scatter(xValues, yValues, color=currentColor)
        plt.plot(xValues, yValues, color=currentColor, drawstyle='steps-post')
        i+=1
        f1 = np.array(xValues)
        f2 = np.array(yValues)
        AUCs.append((np.sum(np.abs(np.diff(f1))*f2[:-1])))

    plt.xlabel("False Positive")
    plt.ylabel("False Negative")
    plt.title("Pareto Front")
    plt.ylim(bottom=0.0)
    plt.xlim(left=0.0)
    plt.show()

    print(AUCs)
