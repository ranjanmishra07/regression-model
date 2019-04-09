from numpy import *

def compute_error(b,m,points):
    total_error=0
    for i in range(0,len(points)):
        x = points[i,0]
        y= points[i,1]
        total_error += (y-(m*x+b)) **2
    return total_error/float(len(points))

def step_gradient(b_current,m_current,points,learning_rate):
    #gradient descent
    b_gradient= 0
    m_gradient= 0
    N= float(len(points))
    for i in range(0,len(points)):
        x=points[i,0]
        y=points[i,1]
        b_gradient = -(2/N) * (y-((m_current*x)+b_current))
        m_gradient = -(2/N) * x * (y-((m_current*x)+b_current))
    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)
    return [new_b, new_m]

def gradient_descent_runner(points,starting_b,starting_m,learning_rate,num_iterations):
    m= starting_m
    b= starting_b
    for i in range(num_iterations):
        b , m = step_gradient(b,m,array(points),learning_rate)
    return [b,m]

def run():
    points= genfromtxt('data.csv', delimiter=",")
    initial_b=0
    initial_m=0
    learning_rate=0.001
    num_iterations=1000
    print ("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error(initial_b, initial_m, points)))
    print ("Running...")
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print ("After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, compute_error(b, m, points)))

if __name__ == '__main__':
    run()