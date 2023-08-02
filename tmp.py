def is_Sum_Of_Powers_Of_Two ( n ) : 
    if ( n < 0 ) : 
        return False 
        
    
    sum1 = 0 
    
    sum2 = 0 
    
    for i in range ( 1 , n + 1 ) : 
        if ( ( n [ i ] & 1 ) == 0 ) : 
            sum1 = sum1 + sum2 
            
            sum2 = sum2 + i 
            
            
        
        
    
    if ( sum1 == sum2 ) : 
        return True 
        
    else : 
        return False 
        
    
    

assert is_Sum_Of_Powers_Of_Two(10) == True
assert is_Sum_Of_Powers_Of_Two(7) == False
assert is_Sum_Of_Powers_Of_Two(14) == True
