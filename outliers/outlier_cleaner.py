#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    errors = zip(ages, net_worths, map(lambda x: (x[0] - x[1])**2, zip(predictions, net_worths)))
    cleaned_data = sorted(errors, key=lambda x: x[2])
    cleaned_data = cleaned_data[:int(len(cleaned_data)*0.90)]

    ### your code goes here

    
    return cleaned_data

