def highest_Power_Of_2 ( n ) : 
    if ( n < 2 ) : 
        return 0 
        ; 
    
    return ( n + 1 ) // 2 
    

assert highest_Power_of_2(10) == 8
assert highest_Power_of_2(19) == 16
assert highest_Power_of_2(32) == 32
