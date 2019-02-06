# Notes

1. model throws an error which seems to be about Q_sigma (transition probability under a policy function which it calculates.) Should recheck all equations
2. There are two action variables -> a_{t+1} and l_t.
3. Need to see if memory management is still an issue. Feels like the most intense part is when the program defines Q and R. Post that, the loops should be fine.
