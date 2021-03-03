#!/usr/bin/env python
# coding: utf-8
#Code written by Emre Uzel for INTL550 Hw1
# In[98]:


from random import uniform


# In[99]:


from functools import wraps

# Function for recording history
def add_to_history(func):
    @wraps(func)
    def wrapper(self, *args, **kw):
        self.history.append(func.__name__)
        return func(self, *args, **kw)
    return wrapper


# In[107]:


class Portfolio(object):
    
    def __init__(self, cash= 0):
        self.cash = cash
        self.current_stock = {}
        self.current_mutfund = {}
        self.current_bond = {}
        self.bond_yearly_payment = {}
        self.history = []
        
    # Everytime sepecified action occur, we add these functions to list history.    
    @add_to_history    
    def addCash(self, val):
        self.cash += val
        
    @add_to_history    
    def buyStock(self,times, stock):
        assert self.cash >= Stock.stock_dict[stock.symbol] * times, "You don't have enough money to buy these stocks."
        if Stock.stock_dict[stock.symbol] in self.current_stock:
            self.current_stock[stock.symbol] += times 
            self.cash -= Stock.stock_dict[stock.symbol] * times
        else:
            self.current_stock[stock.symbol] = times
            self.cash -= stock.stock_dict[stock.symbol] * times
            
    @add_to_history    
    def buyMutualFund(self,share,fund):
        assert self.cash >= share, "You don't have enough money to buy these funds."
        assert fund.name in MutualFund.mut_list, "You can't buy these funds, they are not available."
        if fund.name in self.current_mutfund:
            self.current_mutfund[fund.name] += share 
            self.cash -= share 
        else:
            self.current_mutfund[fund.name] = share 
            self.cash -= share

    @add_to_history    
    def buyBond(self,bond0, price):
        assert self.cash >= price, "You don't have enough cash to buy these bonds"
        assert bond0.name in Bond.bond_list, "You can't buy these bonds, they are not available."
        if bond0.name in self.current_bond:
            self.current_bond[bond0.name] += price 
            self.bond_yearly_payment[bond0.name] += price * bond0.rate / bond0.years
            self.cash -= price
        else:
            self.current_bond[bond0.name] = price
            self.bond_yearly_payment[bond0.name] = price * bond0.rate / bond0.years
            self.cash -= price
            
    
    
    @add_to_history
    def sellStock(self, name, times):
        assert name in self.current_stock, 'There is a problem with selection'
        self.current_stock[name] -= times 
        self.cash = self.cash +  Stock.stock_dict[name] * times * uniform(0.5,1.5)
    
    @add_to_history    
    def sellMutualFund(self,name, share):
        assert name in self.current_mutfund, 'There is a problem with selection'
        assert self.current_mutfund[name] >= share, 'You cannot sell funds you do not have'
        self.current_mutfund[name] -= share
        self.cash  = self.cash + share * uniform(0.9,1.2)
    
    @add_to_history
    def sellBond(self,bond0, price):
        assert bond0.name in self.current_bond, "There is a problem with selection"
        self.current_bond[bond0.name] = self.current_bond[bond0.name] - price
        self.cash = self.cash + price
        self.bond_yearly_payment[bond0.name] = self.bond_yearly_payment[bond0.name] - price * bond0.rate / bond0.years
        
    
    @add_to_history        
    def __str__(self):
        return f"cash: ${self.cash} \nstock: {str(self.current_stock)} \nmutual funds: {str(self.current_mutfund)} \nbonds: {str(self.current_bond)}, yearly payments from bonds: {str(self.bond_yearly_payment)}"
        # Burayı değiştirebilirsin en son.
    
    @add_to_history    
    def withdrawCash(self, with_cash):
        self.cash -= with_cash
        if self.cash <= 0:
            print("This portfolio has no money left to continue investment!\n Please sell some mutual funds or stock options.")
            
        
    


# In[108]:


#Stock class to add or remove stock from portfolio class.
class Stock(object):
    stock_dict = {}
    def __init__(self,price,symbol):
        self.price = price
        self.symbol = symbol
        Stock.stock_dict[symbol] = price
    
    def get_symbol(self):
        return self.symbol


# In[109]:


# Mutual Fund class to add or remove mutual funds from portfolio class.
class MutualFund(object):
    mut_list = []
    def __init__(self,name):
        self.name = name
        MutualFund.mut_list.append(self.name)
    


# In[110]:

#Bond class is created.
class Bond(object):
    bond_list = []
    def __init__(self,name, rate, years):
        self.name = name
        self.rate = rate
        self.years = years
        Bond.bond_list.append(self.name)

    


# In[111]:


# Example shown in homework
#Main implementation
portfolio = Portfolio()
portfolio.addCash(300.50)
s = Stock(20,'HFH')
portfolio.buyStock(5, s)
mf1 = MutualFund("BRT")
mf2 = MutualFund("GHT")
portfolio.buyMutualFund(10.3, mf1)
portfolio.buyMutualFund(2, mf2)
bond0 = Bond("USA", 0.1, 4) #Bond added as 
portfolio.buyBond(bond0, 50)    #Bond bought a certain price 
print(portfolio)
portfolio.sellMutualFund('BRT', 3)
portfolio.sellStock("HFH", 1)
portfolio.withdrawCash(50)
portfolio.sellBond(bond0, 25) #Some of bonds sold.
print(portfolio)
print(" The history of portfolio is:",portfolio.history)



# In[ ]:




