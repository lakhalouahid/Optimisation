#!/usr/bin/env python
# coding: utf-8

# # Description de données par un modèle affine via une descente de gradient

# ### *Enoncé*
# 
# On prend comme cas d'études des données $\boldsymbol{x}$ et $\boldsymbol{y}$ liées par **une relation affine** de paramètres :
# * coefficient directeur : $\boldsymbol{a = 2}$
# * ordonnée à l'origine :  $\boldsymbol{b = 0.5}$
# 
# polluées par un bruit centré obéissant à ***une loi normale d'écart-type 0.2***
# 
# Créer une fonction `DGd` permettant, ***à partir d'un enregistrement de*** $\boldsymbol{N}$ ***couples*** $\boldsymbol{\{\,x_{[i]}\,,\,y_{[i]}\,\}}$, d'obtenir via l'algorithme de **descente de gradient** les paramètres de la relation qui lie ces 2 variables. Cette fonction prendra 5 paramètres d'entrée :
# * **x** : les données $x$
# * **y** : les données $y$
# * **Theta0** : valeur initiale du vecteur de paramètres *(défini comme un np.array)* 
# * **lrate** : gain du gradient *(learning rate)*
# * **Nbmax** : nombre d'itérations maximales

# In[4]:


import numpy as np
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', '')


# In[5]:


theta1 = 2
theta2 = 0.5
bruit = 0.2


# On travaillera tout d'abord sur un premier enregistrement de **11 couples** $\boldsymbol{\{\,x_{[i]}\,,\,y_{[i]}\,\}}$ avec $\boldsymbol{x\in[\,0\,,\,1\,]}$

# In[6]:


x = np.linspace(0,1,11)
y = theta1*x + theta2 + bruit*np.random.randn(len(x))

plt.figure(0,figsize=(12,8))
plt.scatter(x, y, label="données enregistrées")
plt.xlabel('variable x', fontsize=18)
plt.ylabel('variable y', fontsize=18)
plt.legend(fontsize=18)
plt.grid()
plt.show()


# In[7]:


def DGd(x,y,Theta0,lrate,Nbmax):
    N = len(x)
    Theta = np.zeros((2,Nbmax+1))
    Theta[:,0] = Theta0

    for i in range(Nbmax):
        V = y - ( Theta[0,i]*x + Theta[1,i] )
        M = np.hstack( ( -x.reshape(-1,1) , -np.ones((N,1) ) ))
        Theta[:,i+1] = Theta[:,i] - lrate*2*M.T.dot(V)
            
    V = y - ( Theta[0,-1]*x + Theta[1,-1] )
    coutfinal = V.T.dot(V)
    print('Les paramètres de la droite sont a = {:.4f} et b = {:.4f} ]'
          .format(Theta[0,-1],Theta[1,-1]))
    print('La somme des carrés des erreurs de modélisation est : {:.4f}'.format(coutfinal))
    
    plt.figure(1,figsize=(12,8))
    plt.clf()
    plt.scatter(x, y, c='b', label="données enregistrées")
    plt.plot(x,Theta[0,-1]*x + Theta[1,-1], c='r', label="données modélisées")
    plt.xlabel('variable x', fontsize=18)
    plt.ylabel('variable y', fontsize=18)
    plt.legend(fontsize=18)
    plt.grid()
    plt.show()
    
    plt.figure(2,figsize=(12,8))
    plt.clf()
    plt.scatter(np.linspace(0,Nbmax,Nbmax+1), Theta[0,:], c = 'b', label = "theta1 (coef directeur)")
    plt.scatter(np.linspace(0,Nbmax,Nbmax+1), Theta[1,:], c = 'c', label = "theta2 (ordonnée origine)")
    plt.xlabel('itérations', fontsize=18)
    plt.ylabel('paramètres', fontsize=18)
    plt.legend(fontsize=18)
    plt.grid()
    plt.show()


# In[11]:


DGd(x,y,np.array([ 0 , 0 ]),0.05,100)


# On considère ensuite un second enregistrement de **101 couples** $\boldsymbol{\{\,x_{[i]}\,,\,y_{[i]}\,\}}$ avec toujours $\boldsymbol{x\in[\,0\,,\,1\,]}$

# In[12]:


x = np.linspace(0,1,101)
y = theta1*x + theta2 + bruit*np.random.randn(len(x))

plt.figure(0,figsize=(12,8))
plt.clf()
plt.scatter(x, y, label="données enregistrées")
plt.xlabel('variable x', fontsize=18)
plt.ylabel('variable y', fontsize=18)
plt.legend(fontsize=18)
plt.grid()
plt.show()


# In[14]:


DGd(x,y,np.array([ 0 , 0 ]),0.005,100)


# Pour traiter le second jeu de données expérimentales, il a fallu **diviser le gain d'apprentissage par 10**  
# $\boldsymbol{\rightarrow}$ **cela est tout à fait normal**  
# - puisqu'il y a 10 fois plus de données
# - la fonction coût *(qui est la somme des carrés des erreurs)* aura une amplitude 10 fois plus importante
# - et donc il en sera de même pour son gradient *(puisqu'il indique la variation de la fonction coût)*  
# 
# **En conséquence, le gain d'apprentissage doit toujours être normalisé par le nombre de données**, de façon qu'on puisse traiter n'importe quelle taille de jeu de données sans qu'il soit nécessaire de retoucher le gain à chaque fois  
# On prendra également comme crière de performance, non pas la somme des carrés des erreurs, mais le carré de l'erreur moyenne sur une donnée

# In[15]:


def DGd2(x,y,Theta0,lrate,Nbmax):
    N = len(x)
    Theta = np.zeros((2,Nbmax+1))
    Theta[:,0] = Theta0

    for i in range(Nbmax):
        V = y - ( Theta[0,i]*x + Theta[1,i] )
        M = np.hstack( ( -x.reshape(-1,1) , -np.ones((N,1) ) ))
        Theta[:,i+1] = Theta[:,i] - (lrate/N)*2*M.T.dot(V)
            
    V = y - ( Theta[0,-1]*x + Theta[1,-1] )
    coutfinal = (1/N)*V.T.dot(V)
    print('Les paramètres de la droite sont a = {:.4f} et b = {:.4f} ]'
          .format(Theta[0,-1],Theta[1,-1]))
    print('Erreur de modélisation moyenne au carré : {:.4f}'.format(coutfinal))
    
    plt.figure(1,figsize=(12,8))
    plt.clf()
    plt.scatter(x, y, c='b', label="données enregistrées")
    plt.plot(x,Theta[0,-1]*x + Theta[1,-1], c='r', label="données modélisées")
    plt.xlabel('variable x', fontsize=18)
    plt.ylabel('variable y', fontsize=18)
    plt.legend(fontsize=18)
    plt.grid()
    plt.show()
    
    plt.figure(2,figsize=(12,8))
    plt.clf()
    plt.scatter(np.linspace(0,Nbmax,Nbmax+1), Theta[0,:], c = 'b', label = "theta1 (coef directeur)")
    plt.scatter(np.linspace(0,Nbmax,Nbmax+1), Theta[1,:], c = 'c', label = "theta2 (ordonnée origine)")
    plt.xlabel('itérations', fontsize=18)
    plt.ylabel('paramètres', fontsize=18)
    plt.legend(fontsize=18)
    plt.grid()
    plt.show()


# In[16]:


DGd2(x,y,np.array([ 0 , 0 ]),0.5,100)


# On peut ainsi considérer un troisième enregistrement de **1001 couples** $\boldsymbol{\{\,x_{[i]}\,,\,y_{[i]}\,\}}$ avec  $\boldsymbol{x\in[\,0\,,\,1\,]}$ sans avoir à retoucher au gain d'apprentissage

# In[17]:


x = np.linspace(0,1,1001)
y = theta1*x + theta2 + bruit*np.random.randn(len(x))

plt.figure(0,figsize=(12,8))
plt.clf()
plt.scatter(x, y, label="données enregistrées")
plt.xlabel('variable x', fontsize=18)
plt.ylabel('variable y', fontsize=18)
plt.legend(fontsize=18)
plt.grid()
plt.show()


# In[18]:


DGd2(x,y,np.array([ 0 , 0 ]),0.5,100)


# On considère maintenant un quatrième et dernier enregistrement de **1001 couples** $\boldsymbol{\{\,x_{[i]}\,,\,y_{[i]}\,\}}$ avec cette fois $\boldsymbol{x\in[\,0\,,\,10\,]}$

# In[19]:


x = np.linspace(0,10,1001)
y = theta1*x + theta2 + bruit*np.random.randn(len(x))

plt.figure(0,figsize=(12,8))
plt.clf()
plt.scatter(x, y, label="données enregistrées")
plt.xlabel('variable x', fontsize=18)
plt.ylabel('variable y', fontsize=18)
plt.legend(fontsize=18)
plt.grid()
plt.show()


# In[21]:


DGd2(x,y,np.array([ 0 , 0 ]),0.005,100)


# Une nouvelle fois, il a fallu réduire largement le gain d'apprentissage pour obtenir la convergence des paramètres  
# $\boldsymbol{\rightarrow}$  **encore une fois, c'est tout à fait normal**  
# - le gradient de la fonction coût par rapport au paramètre $a$ est la variable $x$
# - la variable $x$ prenant cette fois des valeurs beaucoup plus importantes *(puisque $x$ monte jusque 10 désormais)*, le gradient va augmenter, ce qui oblige à baisser le gain d'apprentissage pour ne pas faire de trop grand pas
# 
# Pour ne pas être tributaire de la valeur brute des paramètres, il est **sain de les normaliser entre 0 et** $\boldsymbol{\pm}1$ avant de lancer la modélisation

# In[22]:


def DGd3(x,y,Theta0,lrate,Nbmax):
    Mx = np.max(x)
    xn = x / Mx
    My = np.max(y)
    yn = y / My
    N = len(x)
    Theta = np.zeros((2,Nbmax+1))
    Theta[:,0] = Theta0
    
    for i in range(Nbmax):
        V = yn - ( Theta[0,i]*xn + Theta[1,i] )
        M = np.hstack( ( -xn.reshape(-1,1) , -np.ones((N,1) ) ))
        Theta[:,i+1] = Theta[:,i] - (lrate/N)*2*M.T.dot(V)
        
    a = Theta[0,-1]*My/Mx
    b = Theta[1,-1]*My
            
    V = y - ( a*x + b )
    coutfinal = (1/N)*V.T.dot(V)    
    print('Les paramètres de la droite sont a = {:.4f} et b = {:.4f}'
          .format(a,b))
    print('Erreur de modélisation moyenne au carré : {:.4f}'.format(coutfinal))
    
    plt.figure(1,figsize=(12,8))
    plt.clf()
    plt.scatter(x, y, c='b', label="données enregistrées")
    plt.plot(x,a*x + b, c='r', label="données modélisées")
    plt.xlabel('variable x', fontsize=18)
    plt.ylabel('variable y', fontsize=18)
    plt.legend(fontsize=18)
    plt.grid()
    plt.show()
    
    plt.figure(2,figsize=(12,8))
    plt.clf()
    plt.scatter(np.linspace(0,Nbmax,Nbmax+1), Theta[0,:]*My/Mx, c = 'b', label = "theta1 (coef directeur)")
    plt.scatter(np.linspace(0,Nbmax,Nbmax+1), Theta[1,:]*My, c = 'c', label = "theta2 (ordonnée origine)")
    plt.xlabel('itérations', fontsize=18)
    plt.ylabel('paramètres', fontsize=18)
    plt.legend(fontsize=18)
    plt.grid()
    plt.show()


# In[23]:


DGd3(x,y,np.array([ 0 , 0 ]),0.5,100)


# Pour alléger le coût du calcul du gradient à chaque itération, on peut se tourner vers **une descente de gradient stochastique**

# In[24]:


def DGd4(x,y,Theta0,lrate,Nbmax,batch):
    
    Mx = np.max(x)
    xn = x / Mx
    My = np.max(y)
    yn = y / My
    
    N = len(x)
    Nbatch = int(N//batch)
    epoch = int(Nbmax//Nbatch+1)
    iterations = epoch*Nbatch
    Theta = np.zeros((2,iterations+1))
    Theta[:,0] = Theta0
    
    print('La taille du batch est :',batch)
    print('donc le nombre de batch pour couvrir le jeu de données est :',Nbatch)
    print('Le nombre de passe pour au moins atteindre le nombre d\'itérations max est :',epoch)
    print('ce qui conduit au total à',iterations,'itérations où l\'on manipule des matrices de dimension : ',
          batch,'x 2')
    
    Data = np.hstack((xn.reshape(-1,1),yn.reshape(-1,1)))
    
    iter = 0
    
    for i in range(epoch):
        np.random.shuffle(Data)
        for j in range(Nbatch):
            V = Data[batch*j:batch*(j+1),1] - ( Theta[0,iter]*Data[batch*j:batch*(j+1),0] + Theta[1,iter] )
            M = np.hstack( ( -Data[batch*j:batch*(j+1),0].reshape(-1,1) , -np.ones((batch,1)) ))
            Theta[:,iter+1] = Theta[:,iter] - (lrate/batch)*2*M.T.dot(V)
            iter += 1
            
    a = Theta[0,-1]*My/Mx
    b = Theta[1,-1]*My
            
    V = y - ( a*x + b )
    coutfinal = (1/N)*V.T.dot(V)
    print('Les paramètres de la droite sont a = {:.4f} et b = {:.4f}'
          .format(a,b))
    print('Erreur de modélisation moyenne au carré : {:.4f}'.format(coutfinal))
    
    plt.figure(1,figsize=(12,8))
    plt.clf()
    plt.scatter(x, y, c='b', label="données enregistrées")
    plt.plot(x,a*x + b, c='r', label="données modélisées")
    plt.xlabel('variable x', fontsize=18)
    plt.ylabel('variable y', fontsize=18)
    plt.legend(fontsize=18)
    plt.grid()
    plt.show()
    
    plt.figure(2,figsize=(12,8))
    plt.clf()
    plt.scatter(np.linspace(0,iterations,iterations+1), Theta[0,:]*My/Mx, c = 'b', label = "theta1 (coef directeur)")
    plt.scatter(np.linspace(0,iterations,iterations+1), Theta[1,:]*My, c = 'c', label = "theta2 (ordonnée origine)")
    plt.xlabel('itérations', fontsize=18)
    plt.ylabel('paramètres', fontsize=18)
    plt.legend(fontsize=18)
    plt.grid()
    plt.show()


# Bien entendu, si ***la taille du batch est égale à la taille des données***, alors les 2 algorithmes de descente de gradient et de descente de gradient stochastique sont identiques

# In[25]:


DGd4(x,y,np.array([ 0 , 0 ]),0.5,100,len(x))


# Ensuite, ***plus on réduit la taille du batch***, moins le coût calculatoire par itération est important, mais plus l'approximation du gradient est bruitée

# In[26]:


DGd4(x,y,np.array([ 0 , 0 ]),0.5,100,100)


# In[27]:


DGd4(x,y,np.array([ 0 , 0 ]),0.5,100,10)


# In[28]:


DGd4(x,y,np.array([ 0 , 0 ]),0.5,100,4)


# In[29]:


DGd4(x,y,np.array([ 0 , 0 ]),0.5,100,1)

