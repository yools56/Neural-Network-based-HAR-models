library(neuralnet)


# r : maximum iteration number needed to converge the coefficients of NN-based HAR models.

HAR_NN.sig <- function(RV.data,i,r,printcoeffi){ 
  
## Data handling for fitting HAR model.
  
Rv.d.y <- as.numeric(as.vector(RV.data[c(2:((nrow(RV.data)-i)+1)),2])) ## from 2 (daily daily time point) ~ (nrow(RV.data)-100)+1 (daily daily time point). 

Rv.d.x1 <- as.numeric(as.vector(RV.data[c(1:(nrow(RV.data)-i)),2])) ## from 1 (daily daily time point) ~ (nrow(RV.data)-100) (daily daily time point). 

## Weekly RV.
Rv.w <- numeric((nrow(RV.data)-i)) 

for(j in 23:(nrow(RV.data)-i)){
  Rv.w[j] <- (Rv.d.x1[j]+Rv.d.x1[j-1]+Rv.d.x1[j-2]+Rv.d.x1[j-3]+Rv.d.x1[j-4])/5
}

## Monthly RV.
Rv.m <- numeric((nrow(RV.data)-i)) 

for(q in 23:(nrow(RV.data)-i)){
  Rv.m[q] <- (Rv.d.x1[q]+Rv.d.x1[q-1]+Rv.d.x1[q-2]+Rv.d.x1[q-3]
                       +Rv.d.x1[q-4]+Rv.d.x1[q-5]+Rv.d.x1[q-6]+Rv.d.x1[q-7]
                       +Rv.d.x1[q-8]+Rv.d.x1[q-9]+Rv.d.x1[q-10]+Rv.d.x1[q-11]
                       +Rv.d.x1[q-12]+Rv.d.x1[q-13]+Rv.d.x1[q-14]+Rv.d.x1[q-15]
                       +Rv.d.x1[q-16]+Rv.d.x1[q-17]+Rv.d.x1[q-18]+Rv.d.x1[q-19]
                       +Rv.d.x1[q-20]+Rv.d.x1[q-21])/22
}


## We need to make the data frame composed of { RV_[t+1d]^(d), RV_[t]^(d), RV_[t]^(w), RV_[t]^(m) } to fit the HAR model to the data.

Rv.d.y2 <- Rv.d.y[23:(nrow(RV.data)-i)] 
Rv.d.x1.2 <- Rv.d.x1[23:(nrow(RV.data)-i)]
Rv.w.2 <- Rv.w[23:(nrow(RV.data)-i)]
Rv.m.2 <- Rv.m[23:(nrow(RV.data)-i)]


## Firstly, cbind { Rv_[t+1d]^(d), Rv[t]^(d), Rv[t]^(w), Rv[t]^(m) }.
tomorrow.today.week.month.data.frame <- cbind(Rv.d.y2,Rv.d.x1.2,Rv.w.2,Rv.m.2) 

## transform to the data frame. 
tomorrow.today.week.month.data.frame <-as.data.frame(tomorrow.today.week.month.data.frame) 
colnames(tomorrow.today.week.month.data.frame) <- c("Rv(d)(t+1d)","Rv(d)(t)","Rv(w)(t)","Rv(m)(t)")
## tomorrow.today.week.month.data.frame : A complete data set to run a regression model with 22 points removed.
## Since there are a lot of data, there is no serious problem even if the data from 1th to 22th daily daily time point are removed.


###########  STEP 1 : initial HAR model fitting to data set.

initial.HAR.model.fit <- lm(Rv.d.y2~Rv.d.x1.2+Rv.w.2+Rv.m.2)   

initial.HAR.model.fit.coeffi <- initial.HAR.model.fit$coefficients

initial.HAR.model.fit.coeffi <- as.vector(initial.HAR.model.fit.coeffi)

## In here, our idea is fitting the single hidden layer feedforward NN model (with 'q' : 5) to
## residuals obtained from fitting HAR model to data set 
## with Rv.d.x1.2(=Rv(t)(d)),Rv.w.2(=Rv(t)(w)),Rv.m.2(=Rv(t)(m)).


## Definition of residual data set (Note that when we set the residual set, we need to remove constant term estimated from fitting HAR model to KOSPI RV series.)
## That is, residual in this part is defined as follow : RV_[t+1d]^(d) - ( beta(d)^(hat)*RV_[t]^(d) + beta(w)^(hat)*RV_[t]^(w) + beta(m)^(hat)*RV_[t]^(m) ).
## Why we define the residual data set like above explanation is to make sure that the identifiability of constant term in the NN based HAR models.

## ¡Ø Of course, there is only one constant term in the HAR-NN model, but strictly speaking, two constant terms obtained by two different model fits (HAR & NN, respectively) are added together to form one constant term.

## initial.res : RV_[t+1d]^(d) - ( beta(d)^(hat)*RV_[t]^(d) + beta(w)^(hat)*RV_[t]^(w) + beta(m)^(hat)*RV_[t]^(m) ) 
initial.res <- Rv.d.y2-(initial.HAR.model.fit.coeffi[2]*Rv.d.x1.2+initial.HAR.model.fit.coeffi[3]*Rv.w.2+initial.HAR.model.fit.coeffi[4]*Rv.m.2)

initial.res.today.week.month.data.frame<- cbind(initial.res,Rv.d.x1.2,Rv.w.2,Rv.m.2)  
## Above data set is composed of (initial.res, Rv(t)(d), Rv(t)(w), Rv(t)(m) ) 

initial.res.today.week.month.data.frame <-as.data.frame(initial.res.today.week.month.data.frame)
colnames(initial.res.today.week.month.data.frame) <- c("initial.res","Rv(d)(t)","Rv(w)(t)","Rv(m)(t)")


###########  STEP 2 : Fitting the single hidden layer feedforward NN model (with 'q' : 5) to
###########           initial residuals obtained from STEP 1.

initial.NN.model.fit.to.initial.res <- neuralnet(initial.res~Rv.d.x1.2+Rv.w.2+Rv.m.2,data=initial.res.today.week.month.data.frame,
                                                 hidden=5,linear.output=T)


## Coefficent values for RV_[t]^(d) (numeric vector for storing updated values.).
HAR.model.coeffi.beta_d.vector <- numeric(r)
u.HAR.model.coeffi.beta_d.vector <- numeric(r) # 'u' : updated.

## Coefficent values for RV_[t]^(w) (numeric vector for storing updated values.).
HAR.model.coeffi.beta_w.vector <- numeric(r)
u.HAR.model.coeffi.beta_w.vector <- numeric(r)

## Coefficent values for RV_[t]^(m) (numeric vector for storing updated values.).
HAR.model.coeffi.beta_m.vector <- numeric(r)
u.HAR.model.coeffi.beta_m.vector <- numeric(r)

###########  STEP 3 : Fit the HAR model again to the [ Rv_[t+1d]^(d) - fitted values obtained from step 2 ].

for(p in 1:r){
  
  ## Re-fit the HAR model to the [ Rv_[t+1d]^(d) - fitted values obtained from step 2 ].
  refit.HAR.model <- lm((Rv.d.y2 - initial.NN.model.fit.to.initial.res$net.result[[1]])~Rv.d.x1.2+Rv.w.2+Rv.m.2)
  
  ## Coefficient vector corresponding to the HAR model part in the HAR-NN model with respect to the above refitted HAR model's coefficients.
  
  HAR.model.coeffi.refitted <-(refit.HAR.model$coefficients)
  HAR.model.coeffi.refitted <- as.vector(HAR.model.coeffi.refitted)
  
  
  
  HAR.model.coeffi.beta_d.vector[p] <- HAR.model.coeffi.refitted[2] 
  
  HAR.model.coeffi.beta_w.vector[p] <- HAR.model.coeffi.refitted[3]
  
  HAR.model.coeffi.beta_m.vector[p] <- HAR.model.coeffi.refitted[4]
  
########### STEP 4 : Fit the single hidden layer feedforward NN model (with 'q' : 5) again to the
###########          residuals obtained from re-fitting the HAR model in STEP 3. 
  
  ## Definition of res2 data set. (Form is similar to 'initial.res' in STEP 1.)
  res2 <- Rv.d.y2-(HAR.model.coeffi.refitted[2]*Rv.d.x1.2+HAR.model.coeffi.refitted[3]*Rv.w.2+HAR.model.coeffi.refitted[4]*Rv.m.2)
  
  second.res.today.week.month.data.frame<- cbind(res2,Rv.d.x1.2,Rv.w.2,Rv.m.2) 
  ## Above data set is composed of ( res2, Rv(t)(d), Rv(t)(w),Rv(t)(m) ).
  
  second.res.today.week.month.data.frame <-as.data.frame(second.res.today.week.month.data.frame)
  colnames(second.res.today.week.month.data.frame) <- c("res2","Rv(d)(t)","Rv(w)(t)","Rv(m)(t)")
  
  
  
########### Re-Fit the NN model to the residuals (=res2) obtained from re-fitting the HAR model in STEP 3. 
  refit.NN.to.res2 <- neuralnet(res2~Rv.d.x1.2+Rv.w.2+Rv.m.2,data=second.res.today.week.month.data.frame,
                                hidden=5,linear.output=T)
  
  
  
########### Return to STEP 3 : Fit the HAR model again to the [ RV_[t+1d]^(d) - fitted values obtained from step 4 ].
  
  # Re-fit the HAR model to the [ RV_(t+1d)^(d) - fitted values obtained from step 4 ].
  rerefit.HAR <- lm((Rv.d.y2 - refit.NN.to.res2$net.result[[1]])~Rv.d.x1.2+Rv.w.2+Rv.m.2)
  
  ### Coefficient vector corresponding to the HAR model part in the HAR-NN model.
  
  HAR.model.coeffi.rerefitted <-(rerefit.HAR$coefficients)
  HAR.model.coeffi.rerefitted <- as.vector(HAR.model.coeffi.rerefitted)
 
  
  ## Updataed coefficient vectors corresponding to the HAR model part in the HAR-NN model.
  
  u.HAR.model.coeffi.beta_d.vector[p] <- HAR.model.coeffi.rerefitted[2] 
  
  u.HAR.model.coeffi.beta_w.vector[p] <- HAR.model.coeffi.rerefitted[3]
  
  u.HAR.model.coeffi.beta_m.vector[p] <- HAR.model.coeffi.rerefitted[4]
  
  
  ## The convergence condition is set when the absolute difference between the 
  ## updated coefficients of the HAR model part in the HAR-NN model and the coefficients of the HAR model part in the same model 
  ## before being updated is smaller than 0.05 in each coefficient. '0.05' is the limit of the error we set in this real data analyis.
  
  diff1=abs(HAR.model.coeffi.beta_d.vector[p]-u.HAR.model.coeffi.beta_d.vector[p])
  diff2=abs(HAR.model.coeffi.beta_w.vector[p]-u.HAR.model.coeffi.beta_w.vector[p])
  diff3=abs(HAR.model.coeffi.beta_m.vector[p]-u.HAR.model.coeffi.beta_m.vector[p])
  
  if( (diff1<0.05) && (diff2<0.05) && (diff3<0.05) )
  {
    
    
    final.beta_d.coeffi <- as.numeric(u.HAR.model.coeffi.beta_d.vector[p])
    
    final.beta_w.coeffi <- as.numeric(u.HAR.model.coeffi.beta_w.vector[p])
    
    final.beta_m.coeffi <- as.numeric(u.HAR.model.coeffi.beta_m.vector[p])
    
    break
  }
  
########### Repeat the above steps (STEP3-> STEP4-> STEP3) repeatedly until the 
########### coefficients ( Beta(d), Beta(w), Beta(m) ) of the HAR model part in the HAR-NN model converge with respect to above criteria.
  
}


## Final.res data set (This is defined with the use of the converged coefficients of the HAR model part in the HAR-NN model.)
final.res <-(Rv.d.y2-(final.beta_d.coeffi*Rv.d.x1.2+final.beta_w.coeffi*Rv.w.2+final.beta_m.coeffi*Rv.m.2))

 
final.res.today.week.month.data.frame <- cbind(final.res,Rv.d.x1.2,Rv.w.2,Rv.m.2) 
## Above data set is filled with ( final.res, Rv(t)(d), Rv(t)(w), Rv(t)(m) ).
final.res.today.week.month.data.frame <- as.data.frame(final.res.today.week.month.data.frame)
colnames(final.res.today.week.month.data.frame) <- c("final.res","Rv(d)(t)","Rv(w)(t)","Rv(m)(t)")


########### STEP 5 : Fitting the single hidden layer feedforward NN model (with 'q' : 5) lastly to 
###########          the final residuals (final.res) obtained from re-fitting the HAR model in the previous step with the convergent coefficients.
###########          Then, we can get convergent coefficients of the NN model part in the HAR-NN model. 

final.NN.fit.to.final.res <- neuralnet(final.res ~ Rv.d.x1.2+Rv.w.2+Rv.m.2,data=final.res.today.week.month.data.frame,
                                       hidden=5,linear.output=T)

########### After implementing the above line, we can make sure that all of the coefficients in the HAR-NN model can be (globally or locally) converged.


########### sigmoid function values.

a1 <- (1/(1+exp(-(final.NN.fit.to.final.res$weights[[1]][[1]][,1][1] + final.NN.fit.to.final.res$weights[[1]][[1]][,1][2]*as.numeric(RV.data[nrow(RV.data)-i,2]) + final.NN.fit.to.final.res$weights[[1]][[1]][,1][3]*Rv.w[nrow(RV.data)-i]+
                    final.NN.fit.to.final.res$weights[[1]][[1]][,1][4]*Rv.m[nrow(RV.data)-i]))))

a2 <- (1/(1+exp(-(final.NN.fit.to.final.res$weights[[1]][[1]][,2][1] + final.NN.fit.to.final.res$weights[[1]][[1]][,2][2]*as.numeric(RV.data[nrow(RV.data)-i,2]) + final.NN.fit.to.final.res$weights[[1]][[1]][,2][3]*Rv.w[nrow(RV.data)-i]+
                    final.NN.fit.to.final.res$weights[[1]][[1]][,2][4]*Rv.m[nrow(RV.data)-i]))))

a3 <- (1/(1+exp(-(final.NN.fit.to.final.res$weights[[1]][[1]][,3][1] + final.NN.fit.to.final.res$weights[[1]][[1]][,3][2]*as.numeric(RV.data[nrow(RV.data)-i,2]) + final.NN.fit.to.final.res$weights[[1]][[1]][,3][3]*Rv.w[nrow(RV.data)-i]+
                    final.NN.fit.to.final.res$weights[[1]][[1]][,3][4]*Rv.m[nrow(RV.data)-i]))))

a4 <- (1/(1+exp(-(final.NN.fit.to.final.res$weights[[1]][[1]][,4][1] + final.NN.fit.to.final.res$weights[[1]][[1]][,4][2]*as.numeric(RV.data[nrow(RV.data)-i,2]) + final.NN.fit.to.final.res$weights[[1]][[1]][,4][3]*Rv.w[nrow(RV.data)-i]+
                    final.NN.fit.to.final.res$weights[[1]][[1]][,4][4]*Rv.m[nrow(RV.data)-i]))))

a5 <- (1/(1+exp(-(final.NN.fit.to.final.res$weights[[1]][[1]][,5][1] + final.NN.fit.to.final.res$weights[[1]][[1]][,5][2]*as.numeric(RV.data[nrow(RV.data)-i,2]) + final.NN.fit.to.final.res$weights[[1]][[1]][,5][3]*Rv.w[nrow(RV.data)-i]+
                    final.NN.fit.to.final.res$weights[[1]][[1]][,5][4]*Rv.m[nrow(RV.data)-i]))))

## All of the coefficients in the HAR-NN model.


coefficient.list <- list("Beta_0d" = final.beta_d.coeffi, "Beta_0w" = final.beta_w.coeffi, "Beta_0m" = final.beta_m.coeffi,
                "Beta_00" =final.NN.fit.to.final.res$weights[[1]][[2]][[1]],"Beta_1" =final.NN.fit.to.final.res$weights[[1]][[2]][[2]],"gamma_10"=final.NN.fit.to.final.res$weights[[1]][[1]][,1][1],
                "gamma_1d"=final.NN.fit.to.final.res$weights[[1]][[1]][,1][2],"gamma_1w"=final.NN.fit.to.final.res$weights[[1]][[1]][,1][3],"gamma_1m"=final.NN.fit.to.final.res$weights[[1]][[1]][,1][4],
                "Beta_2" =final.NN.fit.to.final.res$weights[[1]][[2]][[3]],"gamma_20"=final.NN.fit.to.final.res$weights[[1]][[1]][,2][1],"gamma_2d"=final.NN.fit.to.final.res$weights[[1]][[1]][,2][2],
                "gamma_2w"=final.NN.fit.to.final.res$weights[[1]][[1]][,2][3],"gamma_2m"=final.NN.fit.to.final.res$weights[[1]][[1]][,2][4],"Beta_3" =final.NN.fit.to.final.res$weights[[1]][[2]][[4]],
                "gamma_30"=final.NN.fit.to.final.res$weights[[1]][[1]][,3][1],"gamma_3d"=final.NN.fit.to.final.res$weights[[1]][[1]][,3][2],"gamma_3w"=final.NN.fit.to.final.res$weights[[1]][[1]][,3][3],
                "gamma_3m"=final.NN.fit.to.final.res$weights[[1]][[1]][,3][4],"Beta_4" =final.NN.fit.to.final.res$weights[[1]][[2]][[5]],"gamma_40"=final.NN.fit.to.final.res$weights[[1]][[1]][,4][1],
                "gamma_4d"=final.NN.fit.to.final.res$weights[[1]][[1]][,4][2],"gamma_4w"=final.NN.fit.to.final.res$weights[[1]][[1]][,4][3],"gamma_4m"=final.NN.fit.to.final.res$weights[[1]][[1]][,4][4],
                "Beta_5" =final.NN.fit.to.final.res$weights[[1]][[2]][[6]],"gamma_50"=final.NN.fit.to.final.res$weights[[1]][[1]][,5][1],"gamma_5d"=final.NN.fit.to.final.res$weights[[1]][[1]][,5][2],
                "gamma_5w"=final.NN.fit.to.final.res$weights[[1]][[1]][,5][3],"gamma_5m"=final.NN.fit.to.final.res$weights[[1]][[1]][,5][4])
                

#### Predicted value for RV_[T+1d]^(d), where T = nrow(RV.data)-i.
predicted.values.for.RV.Tplus1d.d <-final.beta_d.coeffi*as.numeric(RV.data[nrow(RV.data)-i,2])+final.beta_w.coeffi*Rv.w[nrow(RV.data)-i]+final.beta_m.coeffi*Rv.m[nrow(RV.data)-i]+
                                     final.NN.fit.to.final.res$weights[[1]][[2]][[1]]+final.NN.fit.to.final.res$weights[[1]][[2]][[2]]*a1+final.NN.fit.to.final.res$weights[[1]][[2]][[3]]*a2+
                                     final.NN.fit.to.final.res$weights[[1]][[2]][[4]]*a3+final.NN.fit.to.final.res$weights[[1]][[2]][[5]]*a4+final.NN.fit.to.final.res$weights[[1]][[2]][[6]]*a5


if(printcoeffi=="YES"){
  print(coefficient.list)
}

else if(printcoeffi=="NO"){
print(predicted.values.for.RV.Tplus1d.d)
}

else if(printcoeffi=="BOTH"){
  print(coefficient.list)
  print(predicted.values.for.RV.Tplus1d.d)
}


else
  print("You might have a typo error in 'printcoeffi' argument")

}


#HAR_NN.sig(KOSPI.0615.data.RV,100,200,"YES")



######################################################################## 

HAR_NN.tanh <- function(RV.data,i,r,printcoeffi){ 
  
  ## Data handling for fitting HAR model.
  
  Rv.d.y <- as.numeric(as.vector(RV.data[c(2:((nrow(RV.data)-i)+1)),2])) ## from 2 (daily daily time point) ~ (nrow(RV.data)-100)+1 (daily daily time point). 
  
  Rv.d.x1 <- as.numeric(as.vector(RV.data[c(1:(nrow(RV.data)-i)),2])) ## from 1 (daily daily time point) ~ (nrow(RV.data)-100) (daily daily time point). 
  
  ## Weekly RV.
  Rv.w <- numeric((nrow(RV.data)-i)) 
  
  for(j in 23:(nrow(RV.data)-i)){
    Rv.w[j] <- (Rv.d.x1[j]+Rv.d.x1[j-1]+Rv.d.x1[j-2]+Rv.d.x1[j-3]+Rv.d.x1[j-4])/5
  }
  
  ## Monthly RV.
  Rv.m <- numeric((nrow(RV.data)-i)) 
  
  for(q in 23:(nrow(RV.data)-i)){
    Rv.m[q] <- (Rv.d.x1[q]+Rv.d.x1[q-1]+Rv.d.x1[q-2]+Rv.d.x1[q-3]
                +Rv.d.x1[q-4]+Rv.d.x1[q-5]+Rv.d.x1[q-6]+Rv.d.x1[q-7]
                +Rv.d.x1[q-8]+Rv.d.x1[q-9]+Rv.d.x1[q-10]+Rv.d.x1[q-11]
                +Rv.d.x1[q-12]+Rv.d.x1[q-13]+Rv.d.x1[q-14]+Rv.d.x1[q-15]
                +Rv.d.x1[q-16]+Rv.d.x1[q-17]+Rv.d.x1[q-18]+Rv.d.x1[q-19]
                +Rv.d.x1[q-20]+Rv.d.x1[q-21])/22
  }
  
  
  cbind(Rv.w[23:(nrow(RV.data)-i)],Rv.m[23:(nrow(RV.data)-i)])
  
  
  
  
  ## We need to make the data frame composed of { RV_[t+1d]^(d), RV_[t]^(d), RV_[t]^(w), RV_[t]^(m) } to fit the HAR model to the data.
  
  Rv.d.y2 <- Rv.d.y[23:(nrow(RV.data)-i)] 
  Rv.d.x1.2 <- Rv.d.x1[23:(nrow(RV.data)-i)]
  Rv.w.2 <- Rv.w[23:(nrow(RV.data)-i)]
  Rv.m.2 <- Rv.m[23:(nrow(RV.data)-i)]
  
  
  ## Firstly, cbind { Rv_[t+1d]^(d), Rv[t]^(d), Rv[t]^(w), Rv[t]^(m) }.
  tomorrow.today.week.month.data.frame <- cbind(Rv.d.y2,Rv.d.x1.2,Rv.w.2,Rv.m.2) 
  
  ## transform to the data frame. 
  tomorrow.today.week.month.data.frame <-as.data.frame(tomorrow.today.week.month.data.frame) 
  colnames(tomorrow.today.week.month.data.frame) <- c("Rv(d)(t+1d)","Rv(d)(t)","Rv(w)(t)","Rv(m)(t)")
  ## tomorrow.today.week.month.data.frame : A complete data set to run a regression model with 22 points removed.
  ## Since there are a lot of data, there is no serious problem even if the data from 1th to 22th daily daily time point are removed.
  
  
###########  STEP 1 : initial HAR model fitting to data set.
  
  initial.HAR.model.fit<-lm(Rv.d.y2~Rv.d.x1.2+Rv.w.2+Rv.m.2)   
  
  initial.HAR.model.fit.coeffi <- initial.HAR.model.fit$coefficients
  
  initial.HAR.model.fit.coeffi <- as.vector(initial.HAR.model.fit.coeffi)
  
  ## In here, our idea is fitting the single hidden layer feedforward NN model (with 'q' : 10) to
  ## residuals obtained from fitting HAR model to data set 
  ## with Rv.d.x1.2(=Rv(t)(d)),Rv.w.2(=Rv(t)(w)),Rv.m.2(=Rv(t)(m)).
  
  
  ## Definition of residual data set (Note that when we set the residual set, we need to remove constant term estimated from fitting HAR model to KOSPI RV series.)
  ## That is, residual in this part is defined as follow : RV_[t+1d]^(d) - ( beta(d)^(hat)*RV_[t]^(d) + beta(w)^(hat)*RV_[t]^(w) + beta(m)^(hat)*RV_[t]^(m) ).
  ## Why we define the residual data set like above explanation is to make sure that the identifiability of constant term in NN based HAR models.
  
  ## ¡Ø Of course, there is only one constant term in the HAR-NN model, but strictly speaking, two constant terms obtained by two different model fits (HAR & NN, respectively) are added together to form one constant term.
  
  ## initial.res : RV_[t+1d]^(d) - ( beta(d)^(hat)*RV_[t]^(d) + beta(w)^(hat)*RV_[t]^(w) + beta(m)^(hat)*RV_[t]^(m) ) 
  initial.res <- Rv.d.y2-(initial.HAR.model.fit.coeffi[2]*Rv.d.x1.2+initial.HAR.model.fit.coeffi[3]*Rv.w.2+initial.HAR.model.fit.coeffi[4]*Rv.m.2)
  
  initial.res.today.week.month.data.frame<- cbind(initial.res,Rv.d.x1.2,Rv.w.2,Rv.m.2)  
  ## Above data set is composed of ( initial.res, Rv(t)(d), Rv(t)(w), Rv(t)(m) ) 
  
  initial.res.today.week.month.data.frame <-as.data.frame(initial.res.today.week.month.data.frame)
  colnames(initial.res.today.week.month.data.frame) <- c("initial.res","Rv(d)(t)","Rv(w)(t)","Rv(m)(t)")
  
  
###########  STEP 2 : Fitting the single hidden layer feedforward NN model (with 'q' : 10) to
###########           residuals obtained from STEP 1.
  
  initial.NN.model.fit.to.initial.res <- neuralnet(initial.res~Rv.d.x1.2+Rv.w.2+Rv.m.2,data=initial.res.today.week.month.data.frame,
                                                   hidden=10,linear.output=T,act.fct = "tanh")
  
  
  ## Coefficent values for RV_[t]^(d) (numeric vector for storing updated values.).
  HAR.model.coeffi.beta_d.vector <- numeric(r)
  u.HAR.model.coeffi.beta_d.vector <- numeric(r)
  
  ## Coefficent values for RV_[t]^(w) (numeric vector for storing updated values.).
  HAR.model.coeffi.beta_w.vector <- numeric(r)
  u.HAR.model.coeffi.beta_w.vector <- numeric(r)
  
  ## Coefficent values for RV_[t]^(m) (numeric vector for storing updated values.).
  HAR.model.coeffi.beta_m.vector <- numeric(r)
  u.HAR.model.coeffi.beta_m.vector <- numeric(r)
  
###########  STEP 3 : Fit the HAR model again to the [ Rv_[t+1d]^(d) - fitted values obtained from step 2 ].
  
  for(p in 1:r){
    
    ## Re-fit the HAR model to the [ Rv_[t+1d]^(d) - fitted values obtained from step 2 ].
    refit.HAR.model <- lm((Rv.d.y2 - initial.NN.model.fit.to.initial.res$net.result[[1]])~Rv.d.x1.2+Rv.w.2+Rv.m.2)
    
    ## Coefficient vector corresponding to the HAR model part in the HAR-NN model with respect to the above refitted HAR model's coefficients.
    
    HAR.model.coeffi.refitted <-(refit.HAR.model$coefficients)
    HAR.model.coeffi.refitted <- as.vector(HAR.model.coeffi.refitted)
    
    
    HAR.model.coeffi.beta_d.vector[p] <- HAR.model.coeffi.refitted[2] 
    
    HAR.model.coeffi.beta_w.vector[p] <- HAR.model.coeffi.refitted[3]
    
    HAR.model.coeffi.beta_m.vector[p] <- HAR.model.coeffi.refitted[4]
    
########### STEP 4 : Fit the single hidden layer feedforward NN model (with 'q' : 10) again to the
###########          residuals obtained from re-fitting the HAR model in STEP 3. 
    
    ## Definition of res2 data set. (Form is similar to 'initial.res' in STEP 1.)
    res2 <- Rv.d.y2-(HAR.model.coeffi.refitted[2]*Rv.d.x1.2+HAR.model.coeffi.refitted[3]*Rv.w.2+HAR.model.coeffi.refitted[4]*Rv.m.2)
    
    second.res.today.week.month.data.frame<- cbind(res2,Rv.d.x1.2,Rv.w.2,Rv.m.2) 
    ## Above data set is composed of ( res2, Rv(t)(d), Rv(t)(w),Rv(t)(m) ).
    
    second.res.today.week.month.data.frame <-as.data.frame(second.res.today.week.month.data.frame)
    colnames(second.res.today.week.month.data.frame) <- c("res2","Rv(d)(t)","Rv(w)(t)","Rv(m)(t)")
    
    
    
    ###### Re-Fit the NN model to the residuals (=res2) obtained from re-fitting the HAR model in STEP 3. 
    refit.NN.to.res2 <- neuralnet(res2~Rv.d.x1.2+Rv.w.2+Rv.m.2,data=second.res.today.week.month.data.frame,
                                  hidden=10,linear.output=T,act.fct = "tanh")
    
    
    
########### Return to STEP 3 : Fit the HAR model again to the [ RV_[t+1d]^(d) - fitted values obtained from step 4 ].
    
    # Re-fit the HAR model to the [ RV_(t+1d)^(d) - fitted values obtained from step 4 ].
    rerefit.HAR <- lm((Rv.d.y2 - refit.NN.to.res2$net.result[[1]])~Rv.d.x1.2+Rv.w.2+Rv.m.2)
    
    ### Coefficient vector corresponding to the HAR model part in the HAR-NN model.
    
    HAR.model.coeffi.rerefitted <-(rerefit.HAR$coefficients)
    HAR.model.coeffi.rerefitted <- as.vector(HAR.model.coeffi.rerefitted)
    
    
    ## Updataed coefficient vectors corresponding to the HAR model part in the HAR-NN model.
    
    u.HAR.model.coeffi.beta_d.vector[p] <- HAR.model.coeffi.rerefitted[2] 
    
    u.HAR.model.coeffi.beta_w.vector[p] <- HAR.model.coeffi.rerefitted[3]
    
    u.HAR.model.coeffi.beta_m.vector[p] <- HAR.model.coeffi.rerefitted[4]
    
    
    ## The convergence condition is set when the absolute difference between the 
    ## updated coefficients of the HAR model part in the HAR-NN model and the coefficients of the HAR model part in the same model 
    ## before being updated is smaller than 0.05 in each coefficient. '0.05' is the limit of the error we set in this real data analyis.
    
    diff1=abs(HAR.model.coeffi.beta_d.vector[p]-u.HAR.model.coeffi.beta_d.vector[p])
    diff2=abs(HAR.model.coeffi.beta_w.vector[p]-u.HAR.model.coeffi.beta_w.vector[p])
    diff3=abs(HAR.model.coeffi.beta_m.vector[p]-u.HAR.model.coeffi.beta_m.vector[p])
    
    if( (diff1<0.05) && (diff2<0.05) && (diff3<0.05) )
    {
      
      
      final.beta_d.coeffi <- as.numeric(u.HAR.model.coeffi.beta_d.vector[p])
      
      final.beta_w.coeffi <- as.numeric(u.HAR.model.coeffi.beta_w.vector[p])
      
      final.beta_m.coeffi <- as.numeric(u.HAR.model.coeffi.beta_m.vector[p])
      
      break
    }
    
    ########### Repeat the above steps (STEP3-> STEP4-> STEP3) repeatedly until the 
    ########### coefficients ( Beta(d), Beta(w), Beta(m) ) of the HAR model part in the HAR-NN model converge with respect to above criteria.
    
  }
  
  
  ## Final.res data set (This is defined with the use of the converged coefficients of the HAR model part in the HAR-NN model.)
  final.res <-(Rv.d.y2-(final.beta_d.coeffi*Rv.d.x1.2+final.beta_w.coeffi*Rv.w.2+final.beta_m.coeffi*Rv.m.2))
  
  
  final.res.today.week.month.data.frame <- cbind(final.res,Rv.d.x1.2,Rv.w.2,Rv.m.2) 
  ## Above data set is filled with ( final.res, Rv(t)(d), Rv(t)(w), Rv(t)(m) ).
  final.res.today.week.month.data.frame <- as.data.frame(final.res.today.week.month.data.frame)
  colnames(final.res.today.week.month.data.frame) <- c("final.res","Rv(d)(t)","Rv(w)(t)","Rv(m)(t)")
  
  
########### STEP 5 : Fitting the single hidden layer feedforward NN model (with 'q' : 10) lastly to 
###########          the final residuals (final.res) obtained from re-fitting the HAR model in the previous step with the convergent coefficients.
###########          Then, we can get convergent coefficients of the NN model part in the HAR-NN model. 
  
  final.NN.fit.to.final.res <- neuralnet(final.res ~ Rv.d.x1.2+Rv.w.2+Rv.m.2,data=final.res.today.week.month.data.frame,
                                         hidden=10,linear.output=T,act.fct = "tanh")
  
  ########### After implementing the above line, we can make sure that all of the coefficients in the HAR-NN model can be (globally or locally) converged.
  
  
  ########### tanh function values.
  
  a1 <- tanh((final.NN.fit.to.final.res$weights[[1]][[1]][,1][1] + final.NN.fit.to.final.res$weights[[1]][[1]][,1][2]*as.numeric(RV.data[nrow(RV.data)-i,2]) + final.NN.fit.to.final.res$weights[[1]][[1]][,1][3]*Rv.w[nrow(RV.data)-i]+
                final.NN.fit.to.final.res$weights[[1]][[1]][,1][4]*Rv.m[nrow(RV.data)-i]))
  
  a2 <- tanh((final.NN.fit.to.final.res$weights[[1]][[1]][,2][1] + final.NN.fit.to.final.res$weights[[1]][[1]][,2][2]*as.numeric(RV.data[nrow(RV.data)-i,2]) + final.NN.fit.to.final.res$weights[[1]][[1]][,2][3]*Rv.w[nrow(RV.data)-i]+
                final.NN.fit.to.final.res$weights[[1]][[1]][,2][4]*Rv.m[nrow(RV.data)-i]))
  
  a3 <- tanh((final.NN.fit.to.final.res$weights[[1]][[1]][,3][1] + final.NN.fit.to.final.res$weights[[1]][[1]][,3][2]*as.numeric(RV.data[nrow(RV.data)-i,2]) + final.NN.fit.to.final.res$weights[[1]][[1]][,3][3]*Rv.w[nrow(RV.data)-i]+
                final.NN.fit.to.final.res$weights[[1]][[1]][,3][4]*Rv.m[nrow(RV.data)-i]))
  
  a4 <- tanh((final.NN.fit.to.final.res$weights[[1]][[1]][,4][1] + final.NN.fit.to.final.res$weights[[1]][[1]][,4][2]*as.numeric(RV.data[nrow(RV.data)-i,2]) + final.NN.fit.to.final.res$weights[[1]][[1]][,4][3]*Rv.w[nrow(RV.data)-i]+
                final.NN.fit.to.final.res$weights[[1]][[1]][,4][4]*Rv.m[nrow(RV.data)-i]))
  
  a5 <- tanh((final.NN.fit.to.final.res$weights[[1]][[1]][,5][1] + final.NN.fit.to.final.res$weights[[1]][[1]][,5][2]*as.numeric(RV.data[nrow(RV.data)-i,2]) + final.NN.fit.to.final.res$weights[[1]][[1]][,5][3]*Rv.w[nrow(RV.data)-i]+
                final.NN.fit.to.final.res$weights[[1]][[1]][,5][4]*Rv.m[nrow(RV.data)-i]))
  
  a6 <- tanh(final.NN.fit.to.final.res$weights[[1]][[1]][,6][1] + final.NN.fit.to.final.res$weights[[1]][[1]][,6][2]*as.numeric(RV.data[nrow(RV.data)-i,2]) + final.NN.fit.to.final.res$weights[[1]][[1]][,6][3]*Rv.w[nrow(RV.data)-i]+
               final.NN.fit.to.final.res$weights[[1]][[1]][,6][4]*Rv.m[nrow(RV.data)-i])
  
  a7 <- tanh(final.NN.fit.to.final.res$weights[[1]][[1]][,7][1] + final.NN.fit.to.final.res$weights[[1]][[1]][,7][2]*as.numeric(RV.data[nrow(RV.data)-i,2]) + final.NN.fit.to.final.res$weights[[1]][[1]][,7][3]*Rv.w[nrow(RV.data)-i]+
               final.NN.fit.to.final.res$weights[[1]][[1]][,7][4]*Rv.m[nrow(RV.data)-i])
  
  a8 <- tanh(final.NN.fit.to.final.res$weights[[1]][[1]][,8][1] + final.NN.fit.to.final.res$weights[[1]][[1]][,8][2]*as.numeric(RV.data[nrow(RV.data)-i,2]) + final.NN.fit.to.final.res$weights[[1]][[1]][,8][3]*Rv.w[nrow(RV.data)-i]+
               final.NN.fit.to.final.res$weights[[1]][[1]][,8][4]*Rv.m[nrow(RV.data)-i])
  
  a9 <- tanh(final.NN.fit.to.final.res$weights[[1]][[1]][,9][1] + final.NN.fit.to.final.res$weights[[1]][[1]][,9][2]*as.numeric(RV.data[nrow(RV.data)-i,2]) + final.NN.fit.to.final.res$weights[[1]][[1]][,9][3]*Rv.w[nrow(RV.data)-i]+
               final.NN.fit.to.final.res$weights[[1]][[1]][,9][4]*Rv.m[nrow(RV.data)-i])
  
  a10 <- tanh(final.NN.fit.to.final.res$weights[[1]][[1]][,10][1] + final.NN.fit.to.final.res$weights[[1]][[1]][,10][2]*as.numeric(RV.data[nrow(RV.data)-i,2]) + final.NN.fit.to.final.res$weights[[1]][[1]][,10][3]*Rv.w[nrow(RV.data)-i]+
                final.NN.fit.to.final.res$weights[[1]][[1]][,10][4]*Rv.m[nrow(RV.data)-i])
  
  ## All of the coefficients in the HAR-NN model.
  
  coefficient.list <- list("Beta_0d" = final.beta_d.coeffi, "Beta_0w" = final.beta_w.coeffi, "Beta_0m" = final.beta_m.coeffi,
                           "Beta_00" =final.NN.fit.to.final.res$weights[[1]][[2]][[1]],"Beta_1" =final.NN.fit.to.final.res$weights[[1]][[2]][[2]],"gamma_10"=final.NN.fit.to.final.res$weights[[1]][[1]][,1][1],
                           "gamma_1d"=final.NN.fit.to.final.res$weights[[1]][[1]][,1][2],"gamma_1w"=final.NN.fit.to.final.res$weights[[1]][[1]][,1][3],"gamma_1m"=final.NN.fit.to.final.res$weights[[1]][[1]][,1][4],
                           "Beta_2" =final.NN.fit.to.final.res$weights[[1]][[2]][[3]],"gamma_20"=final.NN.fit.to.final.res$weights[[1]][[1]][,2][1],"gamma_2d"=final.NN.fit.to.final.res$weights[[1]][[1]][,2][2],
                           "gamma_2w"=final.NN.fit.to.final.res$weights[[1]][[1]][,2][3],"gamma_2m"=final.NN.fit.to.final.res$weights[[1]][[1]][,2][4],"Beta_3" =final.NN.fit.to.final.res$weights[[1]][[2]][[4]],
                           "gamma_30"=final.NN.fit.to.final.res$weights[[1]][[1]][,3][1],"gamma_3d"=final.NN.fit.to.final.res$weights[[1]][[1]][,3][2],"gamma_3w"=final.NN.fit.to.final.res$weights[[1]][[1]][,3][3],
                           "gamma_3m"=final.NN.fit.to.final.res$weights[[1]][[1]][,3][4],"Beta_4" =final.NN.fit.to.final.res$weights[[1]][[2]][[5]],"gamma_40"=final.NN.fit.to.final.res$weights[[1]][[1]][,4][1],
                           "gamma_4d"=final.NN.fit.to.final.res$weights[[1]][[1]][,4][2],"gamma_4w"=final.NN.fit.to.final.res$weights[[1]][[1]][,4][3],"gamma_4m"=final.NN.fit.to.final.res$weights[[1]][[1]][,4][4],
                           "Beta_5" =final.NN.fit.to.final.res$weights[[1]][[2]][[6]],"gamma_50"=final.NN.fit.to.final.res$weights[[1]][[1]][,5][1],"gamma_5d"=final.NN.fit.to.final.res$weights[[1]][[1]][,5][2],
                           "gamma_5w"=final.NN.fit.to.final.res$weights[[1]][[1]][,5][3],"gamma_5m"=final.NN.fit.to.final.res$weights[[1]][[1]][,5][4],"Beta_6" =final.NN.fit.to.final.res$weights[[1]][[2]][[7]],
                           "gamma_60"=final.NN.fit.to.final.res$weights[[1]][[1]][,6][1],"gamma_6d"=final.NN.fit.to.final.res$weights[[1]][[1]][,6][2],"gamma_6w"=final.NN.fit.to.final.res$weights[[1]][[1]][,6][3],
                           "gamma_6m"=final.NN.fit.to.final.res$weights[[1]][[1]][,6][4],"Beta_7" =final.NN.fit.to.final.res$weights[[1]][[2]][[8]],"gamma_70"=final.NN.fit.to.final.res$weights[[1]][[1]][,7][1],
                           "gamma_7d"=final.NN.fit.to.final.res$weights[[1]][[1]][,7][2],"gamma_7w"=final.NN.fit.to.final.res$weights[[1]][[1]][,7][3],"gamma_7m"=final.NN.fit.to.final.res$weights[[1]][[1]][,7][4],
                           "Beta_8" =final.NN.fit.to.final.res$weights[[1]][[2]][[9]],"gamma_80"=final.NN.fit.to.final.res$weights[[1]][[1]][,8][1],"gamma_8d"=final.NN.fit.to.final.res$weights[[1]][[1]][,8][2],
                           "gamma_8w"=final.NN.fit.to.final.res$weights[[1]][[1]][,8][3],"gamma_8m"=final.NN.fit.to.final.res$weights[[1]][[1]][,8][4],"Beta_9" =final.NN.fit.to.final.res$weights[[1]][[2]][[10]],
                           "gamma_90"=final.NN.fit.to.final.res$weights[[1]][[1]][,9][1],"gamma_9d"=final.NN.fit.to.final.res$weights[[1]][[1]][,9][2],"gamma_9w"=final.NN.fit.to.final.res$weights[[1]][[1]][,9][3],
                           "gamma_9m"=final.NN.fit.to.final.res$weights[[1]][[1]][,9][4],"Beta_10" =final.NN.fit.to.final.res$weights[[1]][[2]][[11]],"gamma_10_0"=final.NN.fit.to.final.res$weights[[1]][[1]][,10][1],
                           "gamma_10d"=final.NN.fit.to.final.res$weights[[1]][[1]][,10][2],"gamma_10w"=final.NN.fit.to.final.res$weights[[1]][[1]][,10][3],"gamma_10m"=final.NN.fit.to.final.res$weights[[1]][[1]][,10][4])
  
  
  #### Predicted value for RV_[T+1d]^(d), where T = nrow(RV.data)-i.
  
  predicted.values.for.RV.Tplus1d.d <- final.beta_d.coeffi*as.numeric(RV.data[nrow(RV.data)-i,2])+final.beta_w.coeffi*Rv.w[nrow(RV.data)-i]+final.beta_m.coeffi*Rv.m[nrow(RV.data)-i]+
    final.NN.fit.to.final.res$weights[[1]][[2]][[1]]+final.NN.fit.to.final.res$weights[[1]][[2]][[2]]*a1+final.NN.fit.to.final.res$weights[[1]][[2]][[3]]*a2+
    final.NN.fit.to.final.res$weights[[1]][[2]][[4]]*a3+final.NN.fit.to.final.res$weights[[1]][[2]][[5]]*a4+final.NN.fit.to.final.res$weights[[1]][[2]][[6]]*a5+
    final.NN.fit.to.final.res$weights[[1]][[2]][[7]]*a6+final.NN.fit.to.final.res$weights[[1]][[2]][[8]]*a7+final.NN.fit.to.final.res$weights[[1]][[2]][[9]]*a8+
    final.NN.fit.to.final.res$weights[[1]][[2]][[10]]*a9+final.NN.fit.to.final.res$weights[[1]][[2]][[11]]*a10
  
  
  if(printcoeffi=="YES"){
    print(coefficient.list)
    }
  
  else if(printcoeffi=="NO"){
    print(predicted.values.for.RV.Tplus1d.d)
  }
  
  else if(printcoeffi=="BOTH"){
    print(coefficient.list)
    print(predicted.values.for.RV.Tplus1d.d)
  }
  
  
  else
    print("You might have a typo error in 'printcoeffi' argument")
  
  
}


#HAR_NN.tanh(KOSPI.0615.data.RV,100,200,"YES")




##### HAR-NN forecasting function.

HAR_NN_forecast <- function(RV.data,i,r,acti.fun){
  
  if(acti.fun=="logistic"){
  
    ########### Predicted value for RV_[T+1d]^(d), where T = nrow(RV.data)-i
    
    HAR_NN.sig(KOSPI.0615.data.RV,i,r,"NO")
  
  }
  else if(acti.fun=="tanh"){
  
    ########### Predicted value for RV_[T+1d]^(d), where T = nrow(RV.data)-i 
    
    HAR_NN.tanh(KOSPI.0615.data.RV,i,r,"NO")
  }
  
    else
      print("You might have a typo error in 'acti.fun' argument")
}


#HAR_NN_forecast(KOSPI.0615.data.RV,100,200,"logistic")

#HAR_NN_forecast(KOSPI.0615.data.RV,100,200,"tanh")

#HAR_NN_forecast(KOSPI.0615.data.RV,100,200,"asdf")






########### HAR-infty-NN model : the second NN-based HAR model. 

########### The optimal q is the same as the above case. (sigmoid=logistic : q=5) & (tanh : q=10) for KOSPI RV series.


HAR_infty_NN.sig <- function(RV.data,i,r,printcoeffi){
  
  Rv.d.y <- as.numeric(as.vector(RV.data[c(2:((nrow(RV.data)-i)+1)),2])) ## from 2 (daily time point) ~ (nrow(RV.data)-100)+1 (daily daily time point). 
  
  Rv.d.x1 <- as.numeric(as.vector(RV.data[c(1:(nrow(RV.data)-i)),2])) ## from 1 (daily time point) ~ (nrow(RV.data)-100) (daily daily time point). 
      
## Data handling for AR(22) model fitting
  
  Rv.d.y2 <- Rv.d.y[23:(nrow(RV.data)-i)]  # (t+1d) daily time point series data
  
  Rv.d.x1.2 <- Rv.d.x1[23:(nrow(RV.data)-i)] # (t) daily time point series data

########### STEP 1 : Initial AR(22) model fitting to data set. 
      
      initial.AR22.model.fit <- arima(Rv.d.y2,order=c(22,0,0),method="CSS") 
      
      initial.AR22.model.coeffi <- initial.AR22.model.fit$coef
      
      initial.AR22.model.coeffi <- as.vector(initial.AR22.model.coeffi)
      
      # (t-1d) daily time point series data
      Rv.d.x1.2.1 <- c(Rv.d.x1[22],Rv.d.x1.2[-length(Rv.d.x1.2)])
      
      # (t-2d) daily time point series data
      Rv.d.x1.2.2 <- c(Rv.d.x1[21],Rv.d.x1[22],Rv.d.x1.2[-(length(Rv.d.x1.2)-1):-length(Rv.d.x1.2)])
      
      # (t-3d) daily time point series data
      Rv.d.x1.2.3 <- c(Rv.d.x1[20],Rv.d.x1[21],Rv.d.x1[22],Rv.d.x1.2[-(length(Rv.d.x1.2)-2):-length(Rv.d.x1.2)])
      
      # (t-4d) daily time point series data
      Rv.d.x1.2.4 <- c(Rv.d.x1[19],Rv.d.x1[20],Rv.d.x1[21],Rv.d.x1[22],Rv.d.x1.2[-(length(Rv.d.x1.2)-3):-length(Rv.d.x1.2)])
      
      # (t-5d) daily time point series data
      Rv.d.x1.2.5 <- c(Rv.d.x1[18],Rv.d.x1[19],Rv.d.x1[20],Rv.d.x1[21],Rv.d.x1[22],Rv.d.x1.2[-(length(Rv.d.x1.2)-4):-length(Rv.d.x1.2)])
      
      # (t-6d) daily time point series data
      Rv.d.x1.2.6 <- c(Rv.d.x1[17],Rv.d.x1[18],Rv.d.x1[19],Rv.d.x1[20],Rv.d.x1[21],Rv.d.x1[22],Rv.d.x1.2[-(length(Rv.d.x1.2)-5):-length(Rv.d.x1.2)])
      
      # (t-7d) daily time point series data
      Rv.d.x1.2.7 <- c(Rv.d.x1[16],Rv.d.x1[17],Rv.d.x1[18],Rv.d.x1[19],Rv.d.x1[20],Rv.d.x1[21],Rv.d.x1[22],Rv.d.x1.2[-(length(Rv.d.x1.2)-6):-length(Rv.d.x1.2)])
      
      # (t-8d) daily time point series data
      Rv.d.x1.2.8 <- c(Rv.d.x1[15],Rv.d.x1[16],Rv.d.x1[17],Rv.d.x1[18],Rv.d.x1[19],Rv.d.x1[20],Rv.d.x1[21],Rv.d.x1[22],Rv.d.x1.2[-(length(Rv.d.x1.2)-7):-length(Rv.d.x1.2)])
      
      # (t-9d) daily time point series data
      Rv.d.x1.2.9 <- c(Rv.d.x1[14],Rv.d.x1[15],Rv.d.x1[16],Rv.d.x1[17],Rv.d.x1[18],Rv.d.x1[19],Rv.d.x1[20],Rv.d.x1[21],Rv.d.x1[22],Rv.d.x1.2[-(length(Rv.d.x1.2)-8):-length(Rv.d.x1.2)])
      
      # (t-10d) daily time point series data
      Rv.d.x1.2.10 <- c(Rv.d.x1[13],Rv.d.x1[14],Rv.d.x1[15],Rv.d.x1[16],Rv.d.x1[17],Rv.d.x1[18],Rv.d.x1[19],Rv.d.x1[20],Rv.d.x1[21],Rv.d.x1[22],Rv.d.x1.2[-(length(Rv.d.x1.2)-9):-length(Rv.d.x1.2)])
      
      # (t-11d) daily time point series data
      Rv.d.x1.2.11 <- c(Rv.d.x1[12],Rv.d.x1[13],Rv.d.x1[14],Rv.d.x1[15],Rv.d.x1[16],Rv.d.x1[17],Rv.d.x1[18],Rv.d.x1[19],Rv.d.x1[20],Rv.d.x1[21],Rv.d.x1[22],Rv.d.x1.2[-(length(Rv.d.x1.2)-10):-length(Rv.d.x1.2)])
      
      # (t-12d) daily time point series data
      Rv.d.x1.2.12 <- c(Rv.d.x1[11],Rv.d.x1[12],Rv.d.x1[13],Rv.d.x1[14],Rv.d.x1[15],Rv.d.x1[16],Rv.d.x1[17],Rv.d.x1[18],Rv.d.x1[19],Rv.d.x1[20],Rv.d.x1[21],Rv.d.x1[22],Rv.d.x1.2[-(length(Rv.d.x1.2)-11):-length(Rv.d.x1.2)])
      
      # (t-13d) daily time point series data
      Rv.d.x1.2.13 <- c(Rv.d.x1[10],Rv.d.x1[11],Rv.d.x1[12],Rv.d.x1[13],Rv.d.x1[14],Rv.d.x1[15],Rv.d.x1[16],Rv.d.x1[17],Rv.d.x1[18],Rv.d.x1[19],Rv.d.x1[20],Rv.d.x1[21],Rv.d.x1[22],Rv.d.x1.2[-(length(Rv.d.x1.2)-12):-length(Rv.d.x1.2)])
      
      # (t-14d) daily time point series data
      Rv.d.x1.2.14 <- c(Rv.d.x1[9],Rv.d.x1[10],Rv.d.x1[11],Rv.d.x1[12],Rv.d.x1[13],Rv.d.x1[14],Rv.d.x1[15],Rv.d.x1[16],Rv.d.x1[17],Rv.d.x1[18],Rv.d.x1[19],Rv.d.x1[20],Rv.d.x1[21],Rv.d.x1[22],Rv.d.x1.2[-(length(Rv.d.x1.2)-13):-length(Rv.d.x1.2)])
      
      # (t-15d) daily time point series data
      Rv.d.x1.2.15 <- c(Rv.d.x1[8],Rv.d.x1[9],Rv.d.x1[10],Rv.d.x1[11],Rv.d.x1[12],Rv.d.x1[13],Rv.d.x1[14],Rv.d.x1[15],Rv.d.x1[16],Rv.d.x1[17],Rv.d.x1[18],Rv.d.x1[19],Rv.d.x1[20],Rv.d.x1[21],Rv.d.x1[22],Rv.d.x1.2[-(length(Rv.d.x1.2)-14):-length(Rv.d.x1.2)])
      
      # (t-16d) daily time point series data
      Rv.d.x1.2.16 <- c(Rv.d.x1[7],Rv.d.x1[8],Rv.d.x1[9],Rv.d.x1[10],Rv.d.x1[11],Rv.d.x1[12],Rv.d.x1[13],Rv.d.x1[14],Rv.d.x1[15],Rv.d.x1[16],Rv.d.x1[17],Rv.d.x1[18],Rv.d.x1[19],Rv.d.x1[20],Rv.d.x1[21],Rv.d.x1[22],Rv.d.x1.2[-(length(Rv.d.x1.2)-15):-length(Rv.d.x1.2)])
      
      # (t-17d) daily time point series data
      Rv.d.x1.2.17 <- c(Rv.d.x1[6],Rv.d.x1[7],Rv.d.x1[8],Rv.d.x1[9],Rv.d.x1[10],Rv.d.x1[11],Rv.d.x1[12],Rv.d.x1[13],Rv.d.x1[14],Rv.d.x1[15],Rv.d.x1[16],Rv.d.x1[17],Rv.d.x1[18],Rv.d.x1[19],Rv.d.x1[20],Rv.d.x1[21],Rv.d.x1[22],Rv.d.x1.2[-(length(Rv.d.x1.2)-16):-length(Rv.d.x1.2)])
      
      # (t-18d) daily time point series data
      Rv.d.x1.2.18 <- c(Rv.d.x1[5],Rv.d.x1[6],Rv.d.x1[7],Rv.d.x1[8],Rv.d.x1[9],Rv.d.x1[10],Rv.d.x1[11],Rv.d.x1[12],Rv.d.x1[13],Rv.d.x1[14],Rv.d.x1[15],Rv.d.x1[16],Rv.d.x1[17],Rv.d.x1[18],Rv.d.x1[19],Rv.d.x1[20],Rv.d.x1[21],Rv.d.x1[22],Rv.d.x1.2[-(length(Rv.d.x1.2)-17):-length(Rv.d.x1.2)])
      
      # (t-19d) daily time point series data
      Rv.d.x1.2.19 <- c(Rv.d.x1[4],Rv.d.x1[5],Rv.d.x1[6],Rv.d.x1[7],Rv.d.x1[8],Rv.d.x1[9],Rv.d.x1[10],Rv.d.x1[11],Rv.d.x1[12],Rv.d.x1[13],Rv.d.x1[14],Rv.d.x1[15],Rv.d.x1[16],Rv.d.x1[17],Rv.d.x1[18],Rv.d.x1[19],Rv.d.x1[20],Rv.d.x1[21],Rv.d.x1[22],Rv.d.x1.2[-(length(Rv.d.x1.2)-18):-length(Rv.d.x1.2)])
      
      # (t-20d) daily time point series data
      Rv.d.x1.2.20 <- c(Rv.d.x1[3],Rv.d.x1[4],Rv.d.x1[5],Rv.d.x1[6],Rv.d.x1[7],Rv.d.x1[8],Rv.d.x1[9],Rv.d.x1[10],Rv.d.x1[11],Rv.d.x1[12],Rv.d.x1[13],Rv.d.x1[14],Rv.d.x1[15],Rv.d.x1[16],Rv.d.x1[17],Rv.d.x1[18],Rv.d.x1[19],Rv.d.x1[20],Rv.d.x1[21],Rv.d.x1[22],Rv.d.x1.2[-(length(Rv.d.x1.2)-19):-length(Rv.d.x1.2)])
      
      # (t-21d) daily time point series data
      Rv.d.x1.2.21 <- c(Rv.d.x1[2],Rv.d.x1[3],Rv.d.x1[4],Rv.d.x1[5],Rv.d.x1[6],Rv.d.x1[7],Rv.d.x1[8],Rv.d.x1[9],Rv.d.x1[10],Rv.d.x1[11],Rv.d.x1[12],Rv.d.x1[13],Rv.d.x1[14],Rv.d.x1[15],Rv.d.x1[16],Rv.d.x1[17],Rv.d.x1[18],Rv.d.x1[19],Rv.d.x1[20],Rv.d.x1[21],Rv.d.x1[22],Rv.d.x1.2[-(length(Rv.d.x1.2)-20):-length(Rv.d.x1.2)])
      
      
      
      ## initial residual
      
      initial.res <-(Rv.d.y2-initial.AR22.model.coeffi[1]*Rv.d.x1.2-initial.AR22.model.coeffi[2]*Rv.d.x1.2.1-initial.AR22.model.coeffi[3]*Rv.d.x1.2.2
                     -initial.AR22.model.coeffi[4]*Rv.d.x1.2.3-initial.AR22.model.coeffi[5]*Rv.d.x1.2.4-initial.AR22.model.coeffi[6]*Rv.d.x1.2.5-initial.AR22.model.coeffi[7]*Rv.d.x1.2.6
                     -initial.AR22.model.coeffi[8]*Rv.d.x1.2.7-initial.AR22.model.coeffi[9]*Rv.d.x1.2.8-initial.AR22.model.coeffi[10]*Rv.d.x1.2.9-initial.AR22.model.coeffi[11]*Rv.d.x1.2.10-initial.AR22.model.coeffi[12]*Rv.d.x1.2.11
                     -initial.AR22.model.coeffi[13]*Rv.d.x1.2.12-initial.AR22.model.coeffi[14]*Rv.d.x1.2.13-initial.AR22.model.coeffi[15]*Rv.d.x1.2.14-initial.AR22.model.coeffi[16]*Rv.d.x1.2.15
                     -initial.AR22.model.coeffi[17]*Rv.d.x1.2.16-initial.AR22.model.coeffi[18]*Rv.d.x1.2.17-initial.AR22.model.coeffi[19]*Rv.d.x1.2.18-initial.AR22.model.coeffi[20]*Rv.d.x1.2.19-initial.AR22.model.coeffi[21]*Rv.d.x1.2.20
                     -initial.AR22.model.coeffi[22]*Rv.d.x1.2.21)
      
      
      RVcombined.data <- cbind(initial.res,Rv.d.x1.2,Rv.d.x1.2.1,Rv.d.x1.2.2,Rv.d.x1.2.3,
                               Rv.d.x1.2.4,Rv.d.x1.2.5,Rv.d.x1.2.6,Rv.d.x1.2.7,Rv.d.x1.2.8
                               ,Rv.d.x1.2.9,Rv.d.x1.2.10,Rv.d.x1.2.11,Rv.d.x1.2.12,Rv.d.x1.2.13
                               ,Rv.d.x1.2.14,Rv.d.x1.2.15,Rv.d.x1.2.16,Rv.d.x1.2.17,Rv.d.x1.2.18
                               ,Rv.d.x1.2.19,Rv.d.x1.2.20,Rv.d.x1.2.21)
      
########### STEP 2 : Initial NN fitting to initial residual set. 
      
      ## NN model fitting to initial res data.
      
      NN.model.fitting.to.initial.res <- neuralnet(initial.res~Rv.d.x1.2+Rv.d.x1.2.1+Rv.d.x1.2.2+Rv.d.x1.2.3+
                                      Rv.d.x1.2.4+Rv.d.x1.2.5+Rv.d.x1.2.6+Rv.d.x1.2.7+Rv.d.x1.2.8
                                    +Rv.d.x1.2.9+Rv.d.x1.2.10+Rv.d.x1.2.11+Rv.d.x1.2.12+Rv.d.x1.2.13
                                    +Rv.d.x1.2.14+Rv.d.x1.2.15+Rv.d.x1.2.16+Rv.d.x1.2.17+Rv.d.x1.2.18
                                    +Rv.d.x1.2.19+Rv.d.x1.2.20+Rv.d.x1.2.21,data=RVcombined.data,
                                    hidden=5,linear.output=T)
      
      
      ## The AR(22) model coefficient value corresponding to RV_[t] (numeric vector to store the updated value)
      AR22.model.coeffi.beta01.vector <- numeric(r) # 
      u.AR22.model.coeffi.beta01.vector <- numeric(r) # u : updated
      
      ## The AR(22) model coefficient value corresponding to RV_[t-1d]
      AR22.model.coeffi.beta02.vector <- numeric(r)
      u.AR22.model.coeffi.beta02.vector <- numeric(r)
      
      ## The same to RV_[t-2d]
      AR22.model.coeffi.beta03.vector <- numeric(r)
      u.AR22.model.coeffi.beta03.vector <- numeric(r)
      
      ## The same to RV_[t-3d]
      AR22.model.coeffi.beta04.vector <- numeric(r)
      u.AR22.model.coeffi.beta04.vector <- numeric(r)
      
      ## The same to RV_[t-4d]
      AR22.model.coeffi.beta05.vector <- numeric(r)
      u.AR22.model.coeffi.beta05.vector <- numeric(r)
      
      ## The same to RV_[t-5d]
      AR22.model.coeffi.beta06.vector <- numeric(r)
      u.AR22.model.coeffi.beta06.vector <- numeric(r)
      
      ## The same to RV_[t-6d]
      AR22.model.coeffi.beta07.vector <- numeric(r)
      u.AR22.model.coeffi.beta07.vector <- numeric(r)
      
      ## The same to RV_[t-7d]
      AR22.model.coeffi.beta08.vector <- numeric(r)
      u.AR22.model.coeffi.beta08.vector <- numeric(r)
      
      ## The same to RV_[t-8d]
      AR22.model.coeffi.beta09.vector <- numeric(r)
      u.AR22.model.coeffi.beta09.vector <- numeric(r)
      
      ## The same to RV_[t-9d]
      AR22.model.coeffi.beta010.vector <- numeric(r)
      u.AR22.model.coeffi.beta010.vector <- numeric(r)
      
      ## The same to RV_[t-10d]
      AR22.model.coeffi.beta011.vector <- numeric(r)
      u.AR22.model.coeffi.beta011.vector <- numeric(r)
      
      ## The same to RV_[t-11d]
      AR22.model.coeffi.beta012.vector <- numeric(r)
      u.AR22.model.coeffi.beta012.vector <- numeric(r)
      
      ## The same to RV_[t-12d]
      AR22.model.coeffi.beta013.vector <- numeric(r)
      u.AR22.model.coeffi.beta013.vector <- numeric(r)
      
      ## The same to RV_[t-13d]
      AR22.model.coeffi.beta014.vector <- numeric(r)
      u.AR22.model.coeffi.beta014.vector <- numeric(r)
      
      ## The same to RV_[t-14d]
      AR22.model.coeffi.beta015.vector <- numeric(r)
      u.AR22.model.coeffi.beta015.vector <- numeric(r)
      
      ## The same to RV_[t-15d]
      AR22.model.coeffi.beta016.vector <- numeric(r)
      u.AR22.model.coeffi.beta016.vector <- numeric(r)
      
      ## The same to RV_[t-16d]
      AR22.model.coeffi.beta017.vector <- numeric(r)
      u.AR22.model.coeffi.beta017.vector <- numeric(r)
      
      ## The same to RV_[t-17d]
      AR22.model.coeffi.beta018.vector <- numeric(r)
      u.AR22.model.coeffi.beta018.vector <- numeric(r)
      
      ## The same to RV_[t-18d]
      AR22.model.coeffi.beta019.vector <- numeric(r)
      u.AR22.model.coeffi.beta019.vector <- numeric(r)
      
      ## The same to RV_[t-19d]
      AR22.model.coeffi.beta020.vector <- numeric(r)
      u.AR22.model.coeffi.beta020.vector <- numeric(r)
      
      ## The same to RV_[t-20d]
      AR22.model.coeffi.beta021.vector <- numeric(r)
      u.AR22.model.coeffi.beta021.vector <- numeric(r)
      
      ## The same to RV_[t-21d]
      AR22.model.coeffi.beta022.vector <- numeric(r)
      u.AR22.model.coeffi.beta022.vector <- numeric(r)
      
      

      
      for(p in 1:r){
        
        
########### STEP 3 : refit.AR22.model to ( Rv_[t+1d](d) - fitted.value of 'NN.model.fitting.to.initial.res' )       
        
        refit.AR22.model <- arima((Rv.d.y2 - NN.model.fitting.to.initial.res$net.result[[1]]),order=c(22,0,0),method="CSS")
        
        ### coefficient
        
        refit.AR22.model.coeffi <-(refit.AR22.model$coef)
        refit.AR22.model.coeffi <- as.vector(refit.AR22.model.coeffi)
        refit.AR22.model.coeffi ## 
        
        
        ## store AR(22) Coefficients obtained from STEP 3
        
        
        AR22.model.coeffi.beta01.vector[p] <- refit.AR22.model.coeffi[1]
              
        AR22.model.coeffi.beta02.vector[p] <- refit.AR22.model.coeffi[2]
       
        AR22.model.coeffi.beta03.vector[p] <- refit.AR22.model.coeffi[3]
        
        AR22.model.coeffi.beta04.vector[p] <- refit.AR22.model.coeffi[4]
        
        AR22.model.coeffi.beta05.vector[p] <- refit.AR22.model.coeffi[5]
        
        AR22.model.coeffi.beta06.vector[p] <- refit.AR22.model.coeffi[6]
        
        AR22.model.coeffi.beta07.vector[p] <- refit.AR22.model.coeffi[7]
        
        AR22.model.coeffi.beta08.vector[p] <- refit.AR22.model.coeffi[8]
       
        AR22.model.coeffi.beta09.vector[p] <- refit.AR22.model.coeffi[9]
        
        AR22.model.coeffi.beta010.vector[p] <- refit.AR22.model.coeffi[10]
        
        AR22.model.coeffi.beta011.vector[p] <- refit.AR22.model.coeffi[11]
       
        AR22.model.coeffi.beta012.vector[p] <- refit.AR22.model.coeffi[12]
       
        AR22.model.coeffi.beta013.vector[p] <- refit.AR22.model.coeffi[13]
       
        AR22.model.coeffi.beta014.vector[p] <- refit.AR22.model.coeffi[14]
       
        AR22.model.coeffi.beta015.vector[p] <- refit.AR22.model.coeffi[15]
       
        AR22.model.coeffi.beta016.vector[p] <- refit.AR22.model.coeffi[16]
        
        AR22.model.coeffi.beta017.vector[p] <- refit.AR22.model.coeffi[17]
       
        AR22.model.coeffi.beta018.vector[p] <- refit.AR22.model.coeffi[18]
        
        AR22.model.coeffi.beta019.vector[p] <- refit.AR22.model.coeffi[19]
       
        AR22.model.coeffi.beta020.vector[p] <- refit.AR22.model.coeffi[20]
       
        AR22.model.coeffi.beta021.vector[p] <- refit.AR22.model.coeffi[21]
       
        AR22.model.coeffi.beta022.vector[p] <- refit.AR22.model.coeffi[22]
       
        
########### STEP 4 : Fit NN again to second.res data.
        
       
        # second.res
        second.res <-(Rv.d.y2-AR22.model.coeffi.beta01.vector[p]*Rv.d.x1.2-AR22.model.coeffi.beta02.vector[p]*Rv.d.x1.2.1-AR22.model.coeffi.beta03.vector[p]*Rv.d.x1.2.2-AR22.model.coeffi.beta04.vector[p]*Rv.d.x1.2.3-AR22.model.coeffi.beta05.vector[p]*Rv.d.x1.2.4-AR22.model.coeffi.beta06.vector[p]*Rv.d.x1.2.5-AR22.model.coeffi.beta07.vector[p]*Rv.d.x1.2.6-AR22.model.coeffi.beta08.vector[p]*Rv.d.x1.2.7
                      -AR22.model.coeffi.beta09.vector[p]*Rv.d.x1.2.8-AR22.model.coeffi.beta010.vector[p]*Rv.d.x1.2.9-AR22.model.coeffi.beta011.vector[p]*Rv.d.x1.2.10-AR22.model.coeffi.beta012.vector[p]*Rv.d.x1.2.11
                      -AR22.model.coeffi.beta013.vector[p]*Rv.d.x1.2.12-AR22.model.coeffi.beta014.vector[p]*Rv.d.x1.2.13-AR22.model.coeffi.beta015.vector[p]*Rv.d.x1.2.14-AR22.model.coeffi.beta016.vector[p]*Rv.d.x1.2.15
                      -AR22.model.coeffi.beta017.vector[p]*Rv.d.x1.2.16-AR22.model.coeffi.beta018.vector[p]*Rv.d.x1.2.17-AR22.model.coeffi.beta019.vector[p]*Rv.d.x1.2.18-AR22.model.coeffi.beta020.vector[p]*Rv.d.x1.2.19-AR22.model.coeffi.beta021.vector[p]*Rv.d.x1.2.20
                      -AR22.model.coeffi.beta022.vector[p]*Rv.d.x1.2.21)
        
        
        RVcombined.data2 <- cbind(second.res,Rv.d.x1.2,Rv.d.x1.2.1,Rv.d.x1.2.2,Rv.d.x1.2.3,
                                  Rv.d.x1.2.4,Rv.d.x1.2.5,Rv.d.x1.2.6,Rv.d.x1.2.7,Rv.d.x1.2.8
                                  ,Rv.d.x1.2.9,Rv.d.x1.2.10,Rv.d.x1.2.11,Rv.d.x1.2.12,Rv.d.x1.2.13
                                  ,Rv.d.x1.2.14,Rv.d.x1.2.15,Rv.d.x1.2.16,Rv.d.x1.2.17,Rv.d.x1.2.18
                                  ,Rv.d.x1.2.19,Rv.d.x1.2.20,Rv.d.x1.2.21)
        
        
        ## re-fit NN model to second.res data.
        refit.NN.model.to.second.res <- neuralnet(second.res~Rv.d.x1.2+Rv.d.x1.2.1+Rv.d.x1.2.2+Rv.d.x1.2.3+
                                        Rv.d.x1.2.4+Rv.d.x1.2.5+Rv.d.x1.2.6+Rv.d.x1.2.7+Rv.d.x1.2.8
                                      +Rv.d.x1.2.9+Rv.d.x1.2.10+Rv.d.x1.2.11+Rv.d.x1.2.12+Rv.d.x1.2.13
                                      +Rv.d.x1.2.14+Rv.d.x1.2.15+Rv.d.x1.2.16+Rv.d.x1.2.17+Rv.d.x1.2.18
                                      +Rv.d.x1.2.19+Rv.d.x1.2.20+Rv.d.x1.2.21,data=RVcombined.data2,
                                      hidden=5,linear.output=T)
        
        
  
############ return to the STEP 3 : rerefit.AR22.model to ( Rv_[t+1d](d) - fitted.value of 'refit.NN.model.to.second.res' ) 
        
        rere.AR22.model.fit <- arima((Rv.d.y2 - refit.NN.model.to.second.res$net.result[[1]]),order=c(22,0,0),method="CSS")
        
        ### coefficient
        
        rere.AR22.model.coeffi <-(rere.AR22.model.fit$coef)
        rere.AR22.model.coeffi <- as.vector(rere.AR22.model.coeffi)
       
        
        
        
        ## store updated AR(22) Coefficients obtained from 'return to the STEP 3'.
        
        u.AR22.model.coeffi.beta01.vector[p] <- rere.AR22.model.coeffi[1] 
        
        u.AR22.model.coeffi.beta02.vector[p] <- rere.AR22.model.coeffi[2] 
        
        u.AR22.model.coeffi.beta03.vector[p] <- rere.AR22.model.coeffi[3] 
        
        u.AR22.model.coeffi.beta04.vector[p] <- rere.AR22.model.coeffi[4] 
        
        u.AR22.model.coeffi.beta05.vector[p] <- rere.AR22.model.coeffi[5] 
        
        u.AR22.model.coeffi.beta06.vector[p] <- rere.AR22.model.coeffi[6] 
        
        u.AR22.model.coeffi.beta07.vector[p] <- rere.AR22.model.coeffi[7] 
        
        u.AR22.model.coeffi.beta08.vector[p] <- rere.AR22.model.coeffi[8] 
        
        u.AR22.model.coeffi.beta09.vector[p] <- rere.AR22.model.coeffi[9] 
        
        u.AR22.model.coeffi.beta010.vector[p] <- rere.AR22.model.coeffi[10] 
        
        u.AR22.model.coeffi.beta011.vector[p] <- rere.AR22.model.coeffi[11] 
        
        u.AR22.model.coeffi.beta012.vector[p] <- rere.AR22.model.coeffi[12] 
        
        u.AR22.model.coeffi.beta013.vector[p] <- rere.AR22.model.coeffi[13] 
        
        u.AR22.model.coeffi.beta014.vector[p] <- rere.AR22.model.coeffi[14] 
        
        u.AR22.model.coeffi.beta015.vector[p] <- rere.AR22.model.coeffi[15] 
        
        u.AR22.model.coeffi.beta016.vector[p] <- rere.AR22.model.coeffi[16] 
        
        u.AR22.model.coeffi.beta017.vector[p] <- rere.AR22.model.coeffi[17] 
        
        u.AR22.model.coeffi.beta018.vector[p] <- rere.AR22.model.coeffi[18] 
        
        u.AR22.model.coeffi.beta019.vector[p] <- rere.AR22.model.coeffi[19] 
        
        u.AR22.model.coeffi.beta020.vector[p] <- rere.AR22.model.coeffi[20] 
        
        u.AR22.model.coeffi.beta021.vector[p] <- rere.AR22.model.coeffi[21] 
        
        u.AR22.model.coeffi.beta022.vector[p] <- rere.AR22.model.coeffi[22] 
        
        
        ## 
        
        diff1=abs(AR22.model.coeffi.beta01.vector[p]-u.AR22.model.coeffi.beta01.vector[p])
        diff2=abs(AR22.model.coeffi.beta02.vector[p]-u.AR22.model.coeffi.beta02.vector[p])
        diff3=abs(AR22.model.coeffi.beta03.vector[p]-u.AR22.model.coeffi.beta03.vector[p])
        diff4=abs(AR22.model.coeffi.beta04.vector[p]-u.AR22.model.coeffi.beta04.vector[p])
        diff5=abs(AR22.model.coeffi.beta05.vector[p]-u.AR22.model.coeffi.beta05.vector[p])
        diff6=abs(AR22.model.coeffi.beta06.vector[p]-u.AR22.model.coeffi.beta06.vector[p])
        diff7=abs(AR22.model.coeffi.beta07.vector[p]-u.AR22.model.coeffi.beta07.vector[p])
        diff8=abs(AR22.model.coeffi.beta08.vector[p]-u.AR22.model.coeffi.beta08.vector[p])
        diff9=abs(AR22.model.coeffi.beta09.vector[p]-u.AR22.model.coeffi.beta09.vector[p])
        diff10=abs(AR22.model.coeffi.beta010.vector[p]-u.AR22.model.coeffi.beta010.vector[p])
        diff11=abs(AR22.model.coeffi.beta011.vector[p]-u.AR22.model.coeffi.beta011.vector[p])
        diff12=abs(AR22.model.coeffi.beta012.vector[p]-u.AR22.model.coeffi.beta012.vector[p])
        diff13=abs(AR22.model.coeffi.beta013.vector[p]-u.AR22.model.coeffi.beta013.vector[p])
        diff14=abs(AR22.model.coeffi.beta014.vector[p]-u.AR22.model.coeffi.beta014.vector[p])
        diff15=abs(AR22.model.coeffi.beta015.vector[p]-u.AR22.model.coeffi.beta015.vector[p])
        diff16=abs(AR22.model.coeffi.beta016.vector[p]-u.AR22.model.coeffi.beta016.vector[p])
        diff17=abs(AR22.model.coeffi.beta017.vector[p]-u.AR22.model.coeffi.beta017.vector[p])
        diff18=abs(AR22.model.coeffi.beta018.vector[p]-u.AR22.model.coeffi.beta018.vector[p])
        diff19=abs(AR22.model.coeffi.beta019.vector[p]-u.AR22.model.coeffi.beta019.vector[p])
        diff20=abs(AR22.model.coeffi.beta020.vector[p]-u.AR22.model.coeffi.beta020.vector[p])
        diff21=abs(AR22.model.coeffi.beta021.vector[p]-u.AR22.model.coeffi.beta021.vector[p])
        diff22=abs(AR22.model.coeffi.beta022.vector[p]-u.AR22.model.coeffi.beta022.vector[p])
        
        
        
        
        if( sum(c(diff1,diff2,diff3,diff4,diff5,diff6,diff7,diff8,diff9,diff10,diff11,diff12,diff13,diff14,diff15,diff16,diff17,diff18,diff19,diff20,diff21,diff22)<0.4)==22 )
        {
         
          # convergent coefficient corresponding to RV_[t] in AR(22) model part of HAR-infty-NN model. 
          AR22.model.t.coeffi <- as.numeric(AR22.model.coeffi.beta01.vector[p])
          
          # The same to RV_[t-1d]
          AR22.model.t.1.coeffi <- as.numeric(AR22.model.coeffi.beta02.vector[p])
          
          # The same to RV_[t-2d]
          AR22.model.t.2.coeffi <- as.numeric(AR22.model.coeffi.beta03.vector[p])
          
          # The same to RV_[t-3d]
          AR22.model.t.3.coeffi <- as.numeric(AR22.model.coeffi.beta04.vector[p])
          
          # The same to RV_[t-4d]
          AR22.model.t.4.coeffi <- as.numeric(AR22.model.coeffi.beta05.vector[p])
          
          # The same to RV_[t-5d]
          AR22.model.t.5.coeffi <- as.numeric(AR22.model.coeffi.beta06.vector[p])
          
          # The same to RV_[t-6d]
          AR22.model.t.6.coeffi <- as.numeric(AR22.model.coeffi.beta07.vector[p])
          
          # The same to RV_[t-7d]
          AR22.model.t.7.coeffi <- as.numeric(AR22.model.coeffi.beta08.vector[p])
          
          # The same to RV_[t-8d]
          AR22.model.t.8.coeffi <- as.numeric(AR22.model.coeffi.beta09.vector[p])
          
          # The same to RV_[t-9d]
          AR22.model.t.9.coeffi <- as.numeric(AR22.model.coeffi.beta010.vector[p])
          
          # The same to RV_[t-10d]
          AR22.model.t.10.coeffi <- as.numeric(AR22.model.coeffi.beta011.vector[p])
          
          # The same to RV_[t-11d]
          AR22.model.t.11.coeffi <- as.numeric(AR22.model.coeffi.beta012.vector[p])
          
          # The same to RV_[t-12d]
          AR22.model.t.12.coeffi <- as.numeric(AR22.model.coeffi.beta013.vector[p])
          
          # The same to RV_[t-13d]
          AR22.model.t.13.coeffi <- as.numeric(AR22.model.coeffi.beta014.vector[p])
          
          # The same to RV_[t-14d]
          AR22.model.t.14.coeffi <- as.numeric(AR22.model.coeffi.beta015.vector[p])
          
          # The same to RV_[t-15d]
          AR22.model.t.15.coeffi <- as.numeric(AR22.model.coeffi.beta016.vector[p])
          
          # The same to RV_[t-16d]
          AR22.model.t.16.coeffi <- as.numeric(AR22.model.coeffi.beta017.vector[p])
          
          # The same to RV_[t-17d]
          AR22.model.t.17.coeffi <- as.numeric(AR22.model.coeffi.beta018.vector[p])
          
          # The same to RV_[t-18d]
          AR22.model.t.18.coeffi <- as.numeric(AR22.model.coeffi.beta019.vector[p])
          
          # The same to RV_[t-19d]
          AR22.model.t.19.coeffi <- as.numeric(AR22.model.coeffi.beta020.vector[p])
          
          # The same to RV_[t-20d]
          AR22.model.t.20.coeffi <- as.numeric(AR22.model.coeffi.beta021.vector[p])
          
          # The same to RV_[t-21d]
          AR22.model.t.21.coeffi <- as.numeric(AR22.model.coeffi.beta022.vector[p])
          
          
          break
        }
        
        ## Repeat the above steps (STEP3-> STEP4-> STEP3) repeatedly 
        ## until the coefficients corresponding to the AR(22) model PART of HAR-infty-NN model 
        ## converge to a certain level. 
        
        
      }
      
      

      
########### STEP 5 : Fit the NN model lastly to the final.res data. 
      
      final.res <-(Rv.d.y2-AR22.model.t.coeffi*Rv.d.x1.2-AR22.model.t.1.coeffi*Rv.d.x1.2.1-AR22.model.t.2.coeffi*Rv.d.x1.2.2-AR22.model.t.3.coeffi*Rv.d.x1.2.3-AR22.model.t.4.coeffi*Rv.d.x1.2.4-AR22.model.t.5.coeffi*Rv.d.x1.2.5-AR22.model.t.6.coeffi*Rv.d.x1.2.6-AR22.model.t.7.coeffi*Rv.d.x1.2.7
                   -AR22.model.t.8.coeffi*Rv.d.x1.2.8-AR22.model.t.9.coeffi*Rv.d.x1.2.9-AR22.model.t.10.coeffi*Rv.d.x1.2.10-AR22.model.t.11.coeffi*Rv.d.x1.2.11
                   -AR22.model.t.12.coeffi*Rv.d.x1.2.12-AR22.model.t.13.coeffi*Rv.d.x1.2.13-AR22.model.t.14.coeffi*Rv.d.x1.2.14-AR22.model.t.15.coeffi*Rv.d.x1.2.15
                   -AR22.model.t.16.coeffi*Rv.d.x1.2.16-AR22.model.t.17.coeffi*Rv.d.x1.2.17-AR22.model.t.18.coeffi*Rv.d.x1.2.18-AR22.model.t.19.coeffi*Rv.d.x1.2.19-AR22.model.t.20.coeffi*Rv.d.x1.2.20
                   -AR22.model.t.21.coeffi*Rv.d.x1.2.21)
      
      ## 
      RVcombined.data3 <- cbind(final.res,Rv.d.x1.2,Rv.d.x1.2.1,Rv.d.x1.2.2,Rv.d.x1.2.3,
                                Rv.d.x1.2.4,Rv.d.x1.2.5,Rv.d.x1.2.6,Rv.d.x1.2.7,Rv.d.x1.2.8
                                ,Rv.d.x1.2.9,Rv.d.x1.2.10,Rv.d.x1.2.11,Rv.d.x1.2.12,Rv.d.x1.2.13
                                ,Rv.d.x1.2.14,Rv.d.x1.2.15,Rv.d.x1.2.16,Rv.d.x1.2.17,Rv.d.x1.2.18
                                ,Rv.d.x1.2.19,Rv.d.x1.2.20,Rv.d.x1.2.21)
      
      
      
      RVcombined.data3 <-as.data.frame(RVcombined.data3)
      
      
      
      NN.model.refitting.to.final.res <- neuralnet(final.res~Rv.d.x1.2+Rv.d.x1.2.1+Rv.d.x1.2.2+Rv.d.x1.2.3+
                                    Rv.d.x1.2.4+Rv.d.x1.2.5+Rv.d.x1.2.6+Rv.d.x1.2.7+Rv.d.x1.2.8
                                  +Rv.d.x1.2.9+Rv.d.x1.2.10+Rv.d.x1.2.11+Rv.d.x1.2.12+Rv.d.x1.2.13
                                  +Rv.d.x1.2.14+Rv.d.x1.2.15+Rv.d.x1.2.16+Rv.d.x1.2.17+Rv.d.x1.2.18
                                  +Rv.d.x1.2.19+Rv.d.x1.2.20+Rv.d.x1.2.21,data=RVcombined.data3,
                                  hidden=5,linear.output=T)
      
################  sigmoid funtion values.
      
      a1 <- (1/(1+exp(-(NN.model.refitting.to.final.res$weights[[1]][[1]][,1][1] + NN.model.refitting.to.final.res$weights[[1]][[1]][,1][2]*Rv.d.x1[nrow(RV.data)-i] + NN.model.refitting.to.final.res$weights[[1]][[1]][,1][3]*Rv.d.x1[nrow(RV.data)-(i+1)]+
                          NN.model.refitting.to.final.res$weights[[1]][[1]][,1][4]*Rv.d.x1[nrow(RV.data)-(i+2)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,1][5]*Rv.d.x1[nrow(RV.data)-(i+3)]+
                          NN.model.refitting.to.final.res$weights[[1]][[1]][,1][6]*Rv.d.x1[nrow(RV.data)-(i+4)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,1][7]*Rv.d.x1[nrow(RV.data)-(i+5)]+
                          +NN.model.refitting.to.final.res$weights[[1]][[1]][,1][8]*Rv.d.x1[nrow(RV.data)-(i+6)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,1][9]*Rv.d.x1[nrow(RV.data)-(i+7)]+
                          NN.model.refitting.to.final.res$weights[[1]][[1]][,1][10]*Rv.d.x1[nrow(RV.data)-(i+8)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,1][11]*Rv.d.x1[nrow(RV.data)-(i+9)]+
                          NN.model.refitting.to.final.res$weights[[1]][[1]][,1][12]*Rv.d.x1[nrow(RV.data)-(i+10)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,1][13]*Rv.d.x1[nrow(RV.data)-(i+11)]+
                          NN.model.refitting.to.final.res$weights[[1]][[1]][,1][14]*Rv.d.x1[nrow(RV.data)-(i+12)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,1][15]*Rv.d.x1[nrow(RV.data)-(i+13)]+
                          NN.model.refitting.to.final.res$weights[[1]][[1]][,1][16]*Rv.d.x1[nrow(RV.data)-(i+14)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,1][17]*Rv.d.x1[nrow(RV.data)-(i+15)]+
                          NN.model.refitting.to.final.res$weights[[1]][[1]][,1][18]*Rv.d.x1[nrow(RV.data)-(i+16)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,1][19]*Rv.d.x1[nrow(RV.data)-(i+17)]+
                          NN.model.refitting.to.final.res$weights[[1]][[1]][,1][20]*Rv.d.x1[nrow(RV.data)-(i+18)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,1][21]*Rv.d.x1[nrow(RV.data)-(i+19)]+
                          NN.model.refitting.to.final.res$weights[[1]][[1]][,1][22]*Rv.d.x1[nrow(RV.data)-(i+20)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,1][23]*Rv.d.x1[nrow(RV.data)-(i+21)]))))
      
      
      a2 <- (1/(1+exp(-(NN.model.refitting.to.final.res$weights[[1]][[1]][,2][1] + NN.model.refitting.to.final.res$weights[[1]][[1]][,2][2]*Rv.d.x1[nrow(RV.data)-i] + NN.model.refitting.to.final.res$weights[[1]][[1]][,2][3]*Rv.d.x1[nrow(RV.data)-(i+1)]+
                          NN.model.refitting.to.final.res$weights[[1]][[1]][,2][4]*Rv.d.x1[nrow(RV.data)-(i+2)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,2][5]*Rv.d.x1[nrow(RV.data)-(i+3)]+
                          NN.model.refitting.to.final.res$weights[[1]][[1]][,2][6]*Rv.d.x1[nrow(RV.data)-(i+4)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,2][7]*Rv.d.x1[nrow(RV.data)-(i+5)]+
                          +NN.model.refitting.to.final.res$weights[[1]][[1]][,2][8]*Rv.d.x1[nrow(RV.data)-(i+6)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,2][9]*Rv.d.x1[nrow(RV.data)-(i+7)]+
                          NN.model.refitting.to.final.res$weights[[1]][[1]][,2][10]*Rv.d.x1[nrow(RV.data)-(i+8)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,2][11]*Rv.d.x1[nrow(RV.data)-(i+9)]+
                          NN.model.refitting.to.final.res$weights[[1]][[1]][,2][12]*Rv.d.x1[nrow(RV.data)-(i+10)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,2][13]*Rv.d.x1[nrow(RV.data)-(i+11)]+
                          NN.model.refitting.to.final.res$weights[[1]][[1]][,2][14]*Rv.d.x1[nrow(RV.data)-(i+12)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,2][15]*Rv.d.x1[nrow(RV.data)-(i+13)]+
                          NN.model.refitting.to.final.res$weights[[1]][[1]][,2][16]*Rv.d.x1[nrow(RV.data)-(i+14)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,2][17]*Rv.d.x1[nrow(RV.data)-(i+15)]+
                          NN.model.refitting.to.final.res$weights[[1]][[1]][,2][18]*Rv.d.x1[nrow(RV.data)-(i+16)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,2][19]*Rv.d.x1[nrow(RV.data)-(i+17)]+
                          NN.model.refitting.to.final.res$weights[[1]][[1]][,2][20]*Rv.d.x1[nrow(RV.data)-(i+18)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,2][21]*Rv.d.x1[nrow(RV.data)-(i+19)]+
                          NN.model.refitting.to.final.res$weights[[1]][[1]][,2][22]*Rv.d.x1[nrow(RV.data)-(i+20)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,2][23]*Rv.d.x1[nrow(RV.data)-(i+21)]))))

      a3 <- (1/(1+exp(-(NN.model.refitting.to.final.res$weights[[1]][[1]][,3][1] + NN.model.refitting.to.final.res$weights[[1]][[1]][,3][2]*Rv.d.x1[nrow(RV.data)-i] + NN.model.refitting.to.final.res$weights[[1]][[1]][,3][3]*Rv.d.x1[nrow(RV.data)-(i+1)]+
                          NN.model.refitting.to.final.res$weights[[1]][[1]][,3][4]*Rv.d.x1[nrow(RV.data)-(i+2)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,3][5]*Rv.d.x1[nrow(RV.data)-(i+3)]+
                          NN.model.refitting.to.final.res$weights[[1]][[1]][,3][6]*Rv.d.x1[nrow(RV.data)-(i+4)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,3][7]*Rv.d.x1[nrow(RV.data)-(i+5)]+
                          +NN.model.refitting.to.final.res$weights[[1]][[1]][,3][8]*Rv.d.x1[nrow(RV.data)-(i+6)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,3][9]*Rv.d.x1[nrow(RV.data)-(i+7)]+
                          NN.model.refitting.to.final.res$weights[[1]][[1]][,3][10]*Rv.d.x1[nrow(RV.data)-(i+8)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,3][11]*Rv.d.x1[nrow(RV.data)-(i+9)]+
                          NN.model.refitting.to.final.res$weights[[1]][[1]][,3][12]*Rv.d.x1[nrow(RV.data)-(i+10)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,3][13]*Rv.d.x1[nrow(RV.data)-(i+11)]+
                          NN.model.refitting.to.final.res$weights[[1]][[1]][,3][14]*Rv.d.x1[nrow(RV.data)-(i+12)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,3][15]*Rv.d.x1[nrow(RV.data)-(i+13)]+
                          NN.model.refitting.to.final.res$weights[[1]][[1]][,3][16]*Rv.d.x1[nrow(RV.data)-(i+14)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,3][17]*Rv.d.x1[nrow(RV.data)-(i+15)]+
                          NN.model.refitting.to.final.res$weights[[1]][[1]][,3][18]*Rv.d.x1[nrow(RV.data)-(i+16)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,3][19]*Rv.d.x1[nrow(RV.data)-(i+17)]+
                          NN.model.refitting.to.final.res$weights[[1]][[1]][,3][20]*Rv.d.x1[nrow(RV.data)-(i+18)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,3][21]*Rv.d.x1[nrow(RV.data)-(i+19)]+
                          NN.model.refitting.to.final.res$weights[[1]][[1]][,3][22]*Rv.d.x1[nrow(RV.data)-(i+20)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,3][23]*Rv.d.x1[nrow(RV.data)-(i+21)]))))
      
      a4 <- (1/(1+exp(-(NN.model.refitting.to.final.res$weights[[1]][[1]][,4][1] + NN.model.refitting.to.final.res$weights[[1]][[1]][,4][2]*Rv.d.x1[nrow(RV.data)-i] + NN.model.refitting.to.final.res$weights[[1]][[1]][,4][3]*Rv.d.x1[nrow(RV.data)-(i+1)]+
                          NN.model.refitting.to.final.res$weights[[1]][[1]][,4][4]*Rv.d.x1[nrow(RV.data)-(i+2)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,4][5]*Rv.d.x1[nrow(RV.data)-(i+3)]+
                          NN.model.refitting.to.final.res$weights[[1]][[1]][,4][6]*Rv.d.x1[nrow(RV.data)-(i+4)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,4][7]*Rv.d.x1[nrow(RV.data)-(i+5)]+
                          +NN.model.refitting.to.final.res$weights[[1]][[1]][,4][8]*Rv.d.x1[nrow(RV.data)-(i+6)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,4][9]*Rv.d.x1[nrow(RV.data)-(i+7)]+
                          NN.model.refitting.to.final.res$weights[[1]][[1]][,4][10]*Rv.d.x1[nrow(RV.data)-(i+8)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,4][11]*Rv.d.x1[nrow(RV.data)-(i+9)]+
                          NN.model.refitting.to.final.res$weights[[1]][[1]][,4][12]*Rv.d.x1[nrow(RV.data)-(i+10)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,4][13]*Rv.d.x1[nrow(RV.data)-(i+11)]+
                          NN.model.refitting.to.final.res$weights[[1]][[1]][,4][14]*Rv.d.x1[nrow(RV.data)-(i+12)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,4][15]*Rv.d.x1[nrow(RV.data)-(i+13)]+
                          NN.model.refitting.to.final.res$weights[[1]][[1]][,4][16]*Rv.d.x1[nrow(RV.data)-(i+14)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,4][17]*Rv.d.x1[nrow(RV.data)-(i+15)]+
                          NN.model.refitting.to.final.res$weights[[1]][[1]][,4][18]*Rv.d.x1[nrow(RV.data)-(i+16)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,4][19]*Rv.d.x1[nrow(RV.data)-(i+17)]+
                          NN.model.refitting.to.final.res$weights[[1]][[1]][,4][20]*Rv.d.x1[nrow(RV.data)-(i+18)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,4][21]*Rv.d.x1[nrow(RV.data)-(i+19)]+
                          NN.model.refitting.to.final.res$weights[[1]][[1]][,4][22]*Rv.d.x1[nrow(RV.data)-(i+20)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,4][23]*Rv.d.x1[nrow(RV.data)-(i+21)]))))
      
      
      a5 <- (1/(1+exp(-(NN.model.refitting.to.final.res$weights[[1]][[1]][,5][1] + NN.model.refitting.to.final.res$weights[[1]][[1]][,5][2]*Rv.d.x1[nrow(RV.data)-i] + NN.model.refitting.to.final.res$weights[[1]][[1]][,5][3]*Rv.d.x1[nrow(RV.data)-(i+1)]+
                          NN.model.refitting.to.final.res$weights[[1]][[1]][,5][4]*Rv.d.x1[nrow(RV.data)-(i+2)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,5][5]*Rv.d.x1[nrow(RV.data)-(i+3)]+
                          NN.model.refitting.to.final.res$weights[[1]][[1]][,5][6]*Rv.d.x1[nrow(RV.data)-(i+4)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,5][7]*Rv.d.x1[nrow(RV.data)-(i+5)]+
                          +NN.model.refitting.to.final.res$weights[[1]][[1]][,5][8]*Rv.d.x1[nrow(RV.data)-(i+6)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,5][9]*Rv.d.x1[nrow(RV.data)-(i+7)]+
                          NN.model.refitting.to.final.res$weights[[1]][[1]][,5][10]*Rv.d.x1[nrow(RV.data)-(i+8)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,5][11]*Rv.d.x1[nrow(RV.data)-(i+9)]+
                          NN.model.refitting.to.final.res$weights[[1]][[1]][,5][12]*Rv.d.x1[nrow(RV.data)-(i+10)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,5][13]*Rv.d.x1[nrow(RV.data)-(i+11)]+
                          NN.model.refitting.to.final.res$weights[[1]][[1]][,5][14]*Rv.d.x1[nrow(RV.data)-(i+12)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,5][15]*Rv.d.x1[nrow(RV.data)-(i+13)]+
                          NN.model.refitting.to.final.res$weights[[1]][[1]][,5][16]*Rv.d.x1[nrow(RV.data)-(i+14)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,5][17]*Rv.d.x1[nrow(RV.data)-(i+15)]+
                          NN.model.refitting.to.final.res$weights[[1]][[1]][,5][18]*Rv.d.x1[nrow(RV.data)-(i+16)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,5][19]*Rv.d.x1[nrow(RV.data)-(i+17)]+
                          NN.model.refitting.to.final.res$weights[[1]][[1]][,5][20]*Rv.d.x1[nrow(RV.data)-(i+18)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,5][21]*Rv.d.x1[nrow(RV.data)-(i+19)]+
                          NN.model.refitting.to.final.res$weights[[1]][[1]][,5][22]*Rv.d.x1[nrow(RV.data)-(i+20)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,5][23]*Rv.d.x1[nrow(RV.data)-(i+21)]))))
      
      
## All of the coefficients in the HAR-infty-NN model.
      
      
      coefficient.list <- list("Beta_01" = AR22.model.t.coeffi, "Beta_02" = AR22.model.t.1.coeffi, "Beta_03" = AR22.model.t.2.coeffi,
                               "Beta_04" = AR22.model.t.3.coeffi, "Beta_05" = AR22.model.t.4.coeffi, "Beta_06" = AR22.model.t.5.coeffi,
                               "Beta_07" = AR22.model.t.6.coeffi, "Beta_08" = AR22.model.t.7.coeffi, "Beta_09" = AR22.model.t.8.coeffi,
                               "Beta_010" = AR22.model.t.9.coeffi, "Beta_011" = AR22.model.t.10.coeffi, "Beta_012" = AR22.model.t.11.coeffi,
                               "Beta_013" = AR22.model.t.12.coeffi, "Beta_014" = AR22.model.t.13.coeffi, "Beta_015" = AR22.model.t.14.coeffi,
                               "Beta_016" = AR22.model.t.15.coeffi, "Beta_017" = AR22.model.t.16.coeffi, "Beta_018" = AR22.model.t.17.coeffi,
                               "Beta_019" = AR22.model.t.18.coeffi, "Beta_020" = AR22.model.t.19.coeffi, "Beta_021" = AR22.model.t.20.coeffi,
                               "Beta_022" = AR22.model.t.21.coeffi, "Beta_00" = NN.model.refitting.to.final.res$weights[[1]][[2]][[1]], 
                               
                               "beta_1" = NN.model.refitting.to.final.res$weights[[1]][[2]][[2]], 
                               "gamma_10" = NN.model.refitting.to.final.res$weights[[1]][[1]][,1][1], "gamma_11" = NN.model.refitting.to.final.res$weights[[1]][[1]][,1][2], "gamma_12" = NN.model.refitting.to.final.res$weights[[1]][[1]][,1][3], 
                               "gamma_13" = NN.model.refitting.to.final.res$weights[[1]][[1]][,1][4], "gamma_14" = NN.model.refitting.to.final.res$weights[[1]][[1]][,1][5], "gamma_15" = NN.model.refitting.to.final.res$weights[[1]][[1]][,1][6],
                               "gamma_16" = NN.model.refitting.to.final.res$weights[[1]][[1]][,1][7], "gamma_17" = NN.model.refitting.to.final.res$weights[[1]][[1]][,1][8], "gamma_18" = NN.model.refitting.to.final.res$weights[[1]][[1]][,1][9],
                               "gamma_19" = NN.model.refitting.to.final.res$weights[[1]][[1]][,1][10], "gamma_110" = NN.model.refitting.to.final.res$weights[[1]][[1]][,1][11], "gamma_111" = NN.model.refitting.to.final.res$weights[[1]][[1]][,1][12],
                               "gamma_112" = NN.model.refitting.to.final.res$weights[[1]][[1]][,1][13], "gamma_113" = NN.model.refitting.to.final.res$weights[[1]][[1]][,1][14], "gamma_114" = NN.model.refitting.to.final.res$weights[[1]][[1]][,1][15],
                               "gamma_115" = NN.model.refitting.to.final.res$weights[[1]][[1]][,1][16], "gamma_116" = NN.model.refitting.to.final.res$weights[[1]][[1]][,1][17], "gamma_117" = NN.model.refitting.to.final.res$weights[[1]][[1]][,1][18],
                               "gamma_118" = NN.model.refitting.to.final.res$weights[[1]][[1]][,1][19], "gamma_119" = NN.model.refitting.to.final.res$weights[[1]][[1]][,1][20], "gamma_120" = NN.model.refitting.to.final.res$weights[[1]][[1]][,1][21], 
                               "gamma_121" = NN.model.refitting.to.final.res$weights[[1]][[1]][,1][22], "gamma_122" = NN.model.refitting.to.final.res$weights[[1]][[1]][,1][23],
                               
                               "beta_2" = NN.model.refitting.to.final.res$weights[[1]][[2]][[3]], 
                               "gamma_20" = NN.model.refitting.to.final.res$weights[[1]][[1]][,2][1], "gamma_21" = NN.model.refitting.to.final.res$weights[[1]][[1]][,2][2], "gamma_22" = NN.model.refitting.to.final.res$weights[[1]][[1]][,2][3], 
                               "gamma_23" = NN.model.refitting.to.final.res$weights[[1]][[1]][,2][4], "gamma_24" = NN.model.refitting.to.final.res$weights[[1]][[1]][,2][5], "gamma_25" = NN.model.refitting.to.final.res$weights[[1]][[1]][,2][6],
                               "gamma_26" = NN.model.refitting.to.final.res$weights[[1]][[1]][,2][7], "gamma_27" = NN.model.refitting.to.final.res$weights[[1]][[1]][,2][8], "gamma_28" = NN.model.refitting.to.final.res$weights[[1]][[1]][,2][9],
                               "gamma_29" = NN.model.refitting.to.final.res$weights[[1]][[1]][,2][10], "gamma_210" = NN.model.refitting.to.final.res$weights[[1]][[1]][,2][11], "gamma_211" = NN.model.refitting.to.final.res$weights[[1]][[1]][,2][12],
                               "gamma_212" = NN.model.refitting.to.final.res$weights[[1]][[1]][,2][13], "gamma_213" = NN.model.refitting.to.final.res$weights[[1]][[1]][,2][14], "gamma_214" = NN.model.refitting.to.final.res$weights[[1]][[1]][,2][15],
                               "gamma_215" = NN.model.refitting.to.final.res$weights[[1]][[1]][,2][16], "gamma_216" = NN.model.refitting.to.final.res$weights[[1]][[1]][,2][17], "gamma_217" = NN.model.refitting.to.final.res$weights[[1]][[1]][,2][18],
                               "gamma_218" = NN.model.refitting.to.final.res$weights[[1]][[1]][,2][19], "gamma_219" = NN.model.refitting.to.final.res$weights[[1]][[1]][,2][20], "gamma_220" = NN.model.refitting.to.final.res$weights[[1]][[1]][,2][21], 
                               "gamma_221" = NN.model.refitting.to.final.res$weights[[1]][[1]][,2][22], "gamma_222" = NN.model.refitting.to.final.res$weights[[1]][[1]][,2][23],
                               
                               "beta_3" = NN.model.refitting.to.final.res$weights[[1]][[2]][[4]], 
                               "gamma_30" = NN.model.refitting.to.final.res$weights[[1]][[1]][,3][1], "gamma_31" = NN.model.refitting.to.final.res$weights[[1]][[1]][,3][2], "gamma_32" = NN.model.refitting.to.final.res$weights[[1]][[1]][,3][3], 
                               "gamma_33" = NN.model.refitting.to.final.res$weights[[1]][[1]][,3][4], "gamma_34" = NN.model.refitting.to.final.res$weights[[1]][[1]][,3][5], "gamma_35" = NN.model.refitting.to.final.res$weights[[1]][[1]][,3][6],
                               "gamma_36" = NN.model.refitting.to.final.res$weights[[1]][[1]][,3][7], "gamma_37" = NN.model.refitting.to.final.res$weights[[1]][[1]][,3][8], "gamma_38" = NN.model.refitting.to.final.res$weights[[1]][[1]][,3][9],
                               "gamma_39" = NN.model.refitting.to.final.res$weights[[1]][[1]][,3][10], "gamma_310" = NN.model.refitting.to.final.res$weights[[1]][[1]][,3][11], "gamma_311" = NN.model.refitting.to.final.res$weights[[1]][[1]][,3][12],
                               "gamma_312" = NN.model.refitting.to.final.res$weights[[1]][[1]][,3][13], "gamma_313" = NN.model.refitting.to.final.res$weights[[1]][[1]][,3][14], "gamma_314" = NN.model.refitting.to.final.res$weights[[1]][[1]][,3][15],
                               "gamma_315" = NN.model.refitting.to.final.res$weights[[1]][[1]][,3][16], "gamma_316" = NN.model.refitting.to.final.res$weights[[1]][[1]][,3][17], "gamma_317" = NN.model.refitting.to.final.res$weights[[1]][[1]][,3][18],
                               "gamma_318" = NN.model.refitting.to.final.res$weights[[1]][[1]][,3][19], "gamma_319" = NN.model.refitting.to.final.res$weights[[1]][[1]][,3][20], "gamma_320" = NN.model.refitting.to.final.res$weights[[1]][[1]][,3][21], 
                               "gamma_321" = NN.model.refitting.to.final.res$weights[[1]][[1]][,3][22], "gamma_322" = NN.model.refitting.to.final.res$weights[[1]][[1]][,3][23],
                               
                               "beta_4" = NN.model.refitting.to.final.res$weights[[1]][[2]][[5]], 
                               "gamma_40" = NN.model.refitting.to.final.res$weights[[1]][[1]][,4][1], "gamma_41" = NN.model.refitting.to.final.res$weights[[1]][[1]][,4][2], "gamma_42" = NN.model.refitting.to.final.res$weights[[1]][[1]][,4][3], 
                               "gamma_43" = NN.model.refitting.to.final.res$weights[[1]][[1]][,4][4], "gamma_44" = NN.model.refitting.to.final.res$weights[[1]][[1]][,4][5], "gamma_45" = NN.model.refitting.to.final.res$weights[[1]][[1]][,4][6],
                               "gamma_46" = NN.model.refitting.to.final.res$weights[[1]][[1]][,4][7], "gamma_47" = NN.model.refitting.to.final.res$weights[[1]][[1]][,4][8], "gamma_48" = NN.model.refitting.to.final.res$weights[[1]][[1]][,4][9],
                               "gamma_49" = NN.model.refitting.to.final.res$weights[[1]][[1]][,4][10], "gamma_410" = NN.model.refitting.to.final.res$weights[[1]][[1]][,4][11], "gamma_411" = NN.model.refitting.to.final.res$weights[[1]][[1]][,4][12],
                               "gamma_412" = NN.model.refitting.to.final.res$weights[[1]][[1]][,4][13], "gamma_413" = NN.model.refitting.to.final.res$weights[[1]][[1]][,4][14], "gamma_414" = NN.model.refitting.to.final.res$weights[[1]][[1]][,4][15],
                               "gamma_415" = NN.model.refitting.to.final.res$weights[[1]][[1]][,4][16], "gamma_416" = NN.model.refitting.to.final.res$weights[[1]][[1]][,4][17], "gamma_417" = NN.model.refitting.to.final.res$weights[[1]][[1]][,4][18],
                               "gamma_418" = NN.model.refitting.to.final.res$weights[[1]][[1]][,4][19], "gamma_419" = NN.model.refitting.to.final.res$weights[[1]][[1]][,4][20], "gamma_420" = NN.model.refitting.to.final.res$weights[[1]][[1]][,4][21], 
                               "gamma_421" = NN.model.refitting.to.final.res$weights[[1]][[1]][,4][22], "gamma_422" = NN.model.refitting.to.final.res$weights[[1]][[1]][,4][23],
                               
                               "beta_5" = NN.model.refitting.to.final.res$weights[[1]][[2]][[6]], 
                               "gamma_50" = NN.model.refitting.to.final.res$weights[[1]][[1]][,5][1], "gamma_51" = NN.model.refitting.to.final.res$weights[[1]][[1]][,5][2], "gamma_52" = NN.model.refitting.to.final.res$weights[[1]][[1]][,5][3], 
                               "gamma_53" = NN.model.refitting.to.final.res$weights[[1]][[1]][,5][4], "gamma_54" = NN.model.refitting.to.final.res$weights[[1]][[1]][,5][5], "gamma_55" = NN.model.refitting.to.final.res$weights[[1]][[1]][,5][6],
                               "gamma_56" = NN.model.refitting.to.final.res$weights[[1]][[1]][,5][7], "gamma_57" = NN.model.refitting.to.final.res$weights[[1]][[1]][,5][8], "gamma_58" = NN.model.refitting.to.final.res$weights[[1]][[1]][,5][9],
                               "gamma_59" = NN.model.refitting.to.final.res$weights[[1]][[1]][,5][10], "gamma_510" = NN.model.refitting.to.final.res$weights[[1]][[1]][,5][11], "gamma_511" = NN.model.refitting.to.final.res$weights[[1]][[1]][,5][12],
                               "gamma_512" = NN.model.refitting.to.final.res$weights[[1]][[1]][,5][13], "gamma_513" = NN.model.refitting.to.final.res$weights[[1]][[1]][,5][14], "gamma_514" = NN.model.refitting.to.final.res$weights[[1]][[1]][,5][15],
                               "gamma_515" = NN.model.refitting.to.final.res$weights[[1]][[1]][,5][16], "gamma_516" = NN.model.refitting.to.final.res$weights[[1]][[1]][,5][17], "gamma_517" = NN.model.refitting.to.final.res$weights[[1]][[1]][,5][18],
                               "gamma_518" = NN.model.refitting.to.final.res$weights[[1]][[1]][,5][19], "gamma_519" = NN.model.refitting.to.final.res$weights[[1]][[1]][,5][20], "gamma_520" = NN.model.refitting.to.final.res$weights[[1]][[1]][,5][21], 
                               "gamma_521" = NN.model.refitting.to.final.res$weights[[1]][[1]][,5][22], "gamma_522" = NN.model.refitting.to.final.res$weights[[1]][[1]][,5][23])
                               
      
#### Predicted value for RV_[T+1d]^(d), where T = nrow(RV.data)-i.
      predicted.values.for.RV.Tplus1d.d  <- AR22.model.t.coeffi*Rv.d.x1[nrow(RV.data)-i] + AR22.model.t.1.coeffi*Rv.d.x1[nrow(RV.data)-(i+1)]+
        AR22.model.t.2.coeffi*Rv.d.x1[nrow(RV.data)-(i+2)]+AR22.model.t.3.coeffi*Rv.d.x1[nrow(RV.data)-(i+3)]+AR22.model.t.4.coeffi*Rv.d.x1[nrow(RV.data)-(i+4)]+
      AR22.model.t.5.coeffi*Rv.d.x1[nrow(RV.data)-(i+5)]+AR22.model.t.6.coeffi*Rv.d.x1[nrow(RV.data)-(i+6)]+
        AR22.model.t.7.coeffi*Rv.d.x1[nrow(RV.data)-(i+7)]+AR22.model.t.8.coeffi*Rv.d.x1[nrow(RV.data)-(i+8)]+AR22.model.t.9.coeffi*Rv.d.x1[nrow(RV.data)-(i+9)]
      +AR22.model.t.10.coeffi*Rv.d.x1[nrow(RV.data)-(i+10)]+AR22.model.t.11.coeffi*Rv.d.x1[nrow(RV.data)-(i+11)]+AR22.model.t.12.coeffi*Rv.d.x1[nrow(RV.data)-(i+12)]
      +AR22.model.t.13.coeffi*Rv.d.x1[nrow(RV.data)-(i+13)]+AR22.model.t.14.coeffi*Rv.d.x1[nrow(RV.data)-(i+14)]+AR22.model.t.15.coeffi*Rv.d.x1[nrow(RV.data)-(i+15)]
      +AR22.model.t.16.coeffi*Rv.d.x1[nrow(RV.data)-(i+16)]+AR22.model.t.17.coeffi*Rv.d.x1[nrow(RV.data)-(i+17)]+AR22.model.t.18.coeffi*Rv.d.x1[nrow(RV.data)-(i+18)]
      +AR22.model.t.19.coeffi*Rv.d.x1[nrow(RV.data)-(i+19)]+AR22.model.t.20.coeffi*Rv.d.x1[nrow(RV.data)-(i+20)]
      +AR22.model.t.21.coeffi*Rv.d.x1[nrow(RV.data)-(i+21)]+NN.model.refitting.to.final.res$weights[[1]][[2]][[1]]+NN.model.refitting.to.final.res$weights[[1]][[2]][[2]]*a1+NN.model.refitting.to.final.res$weights[[1]][[2]][[3]]*a2+
        NN.model.refitting.to.final.res$weights[[1]][[2]][[4]]*a3+NN.model.refitting.to.final.res$weights[[1]][[2]][[5]]*a4+NN.model.refitting.to.final.res$weights[[1]][[2]][[6]]*a5
      
      
      if(printcoeffi=="YES"){
        print(coefficient.list)
      }
      
      else if(printcoeffi=="NO"){
        print(predicted.values.for.RV.Tplus1d.d)
      }
      
      else if(printcoeffi=="BOTH"){
        print(coefficient.list)
        print(predicted.values.for.RV.Tplus1d.d)
      }
      
      
      else
        print("You might have a typo error in 'printcoeffi' argument")
      
}
    
  
# HAR_infty_NN.sig(KOSPI.0615.data.RV,100,200,"YES")



#####################################

HAR_infty_NN.tanh <- function(RV.data,i,r,printcoeffi){
  
  Rv.d.y <- as.numeric(as.vector(RV.data[c(2:((nrow(RV.data)-i)+1)),2])) ## from 2 (daily time point) ~ (nrow(RV.data)-100)+1 (daily daily time point). 
  
  Rv.d.x1 <- as.numeric(as.vector(RV.data[c(1:(nrow(RV.data)-i)),2])) ## from 1 (daily time point) ~ (nrow(RV.data)-100) (daily daily time point). 
  
  ## Data handling for AR(22) model fitting
  
  Rv.d.y2 <- Rv.d.y[23:(nrow(RV.data)-i)]  # (t+1d) daily time point series data
  
  Rv.d.x1.2 <- Rv.d.x1[23:(nrow(RV.data)-i)] # (t) daily time point series data
  
########### STEP 1 : Initial AR(22) model fitting to data set. 
  
  initial.AR22.model.fit <- arima(Rv.d.y2,order=c(22,0,0),method="CSS") 
  
  initial.AR22.model.coeffi <- initial.AR22.model.fit$coef
  
  initial.AR22.model.coeffi <- as.vector(initial.AR22.model.coeffi)
  
  # (t-1d) daily time point series data
  Rv.d.x1.2.1 <- c(Rv.d.x1[22],Rv.d.x1.2[-length(Rv.d.x1.2)])
  
  # (t-2d) daily time point series data
  Rv.d.x1.2.2 <- c(Rv.d.x1[21],Rv.d.x1[22],Rv.d.x1.2[-(length(Rv.d.x1.2)-1):-length(Rv.d.x1.2)])
  
  # (t-3d) daily time point series data
  Rv.d.x1.2.3 <- c(Rv.d.x1[20],Rv.d.x1[21],Rv.d.x1[22],Rv.d.x1.2[-(length(Rv.d.x1.2)-2):-length(Rv.d.x1.2)])
  
  # (t-4d) daily time point series data
  Rv.d.x1.2.4 <- c(Rv.d.x1[19],Rv.d.x1[20],Rv.d.x1[21],Rv.d.x1[22],Rv.d.x1.2[-(length(Rv.d.x1.2)-3):-length(Rv.d.x1.2)])
  
  # (t-5d) daily time point series data
  Rv.d.x1.2.5 <- c(Rv.d.x1[18],Rv.d.x1[19],Rv.d.x1[20],Rv.d.x1[21],Rv.d.x1[22],Rv.d.x1.2[-(length(Rv.d.x1.2)-4):-length(Rv.d.x1.2)])
  
  # (t-6d) daily time point series data
  Rv.d.x1.2.6 <- c(Rv.d.x1[17],Rv.d.x1[18],Rv.d.x1[19],Rv.d.x1[20],Rv.d.x1[21],Rv.d.x1[22],Rv.d.x1.2[-(length(Rv.d.x1.2)-5):-length(Rv.d.x1.2)])
  
  # (t-7d) daily time point series data
  Rv.d.x1.2.7 <- c(Rv.d.x1[16],Rv.d.x1[17],Rv.d.x1[18],Rv.d.x1[19],Rv.d.x1[20],Rv.d.x1[21],Rv.d.x1[22],Rv.d.x1.2[-(length(Rv.d.x1.2)-6):-length(Rv.d.x1.2)])
  
  # (t-8d) daily time point series data
  Rv.d.x1.2.8 <- c(Rv.d.x1[15],Rv.d.x1[16],Rv.d.x1[17],Rv.d.x1[18],Rv.d.x1[19],Rv.d.x1[20],Rv.d.x1[21],Rv.d.x1[22],Rv.d.x1.2[-(length(Rv.d.x1.2)-7):-length(Rv.d.x1.2)])
  
  # (t-9d) daily time point series data
  Rv.d.x1.2.9 <- c(Rv.d.x1[14],Rv.d.x1[15],Rv.d.x1[16],Rv.d.x1[17],Rv.d.x1[18],Rv.d.x1[19],Rv.d.x1[20],Rv.d.x1[21],Rv.d.x1[22],Rv.d.x1.2[-(length(Rv.d.x1.2)-8):-length(Rv.d.x1.2)])
  
  # (t-10d) daily time point series data
  Rv.d.x1.2.10 <- c(Rv.d.x1[13],Rv.d.x1[14],Rv.d.x1[15],Rv.d.x1[16],Rv.d.x1[17],Rv.d.x1[18],Rv.d.x1[19],Rv.d.x1[20],Rv.d.x1[21],Rv.d.x1[22],Rv.d.x1.2[-(length(Rv.d.x1.2)-9):-length(Rv.d.x1.2)])
  
  # (t-11d) daily time point series data
  Rv.d.x1.2.11 <- c(Rv.d.x1[12],Rv.d.x1[13],Rv.d.x1[14],Rv.d.x1[15],Rv.d.x1[16],Rv.d.x1[17],Rv.d.x1[18],Rv.d.x1[19],Rv.d.x1[20],Rv.d.x1[21],Rv.d.x1[22],Rv.d.x1.2[-(length(Rv.d.x1.2)-10):-length(Rv.d.x1.2)])
  
  # (t-12d) daily time point series data
  Rv.d.x1.2.12 <- c(Rv.d.x1[11],Rv.d.x1[12],Rv.d.x1[13],Rv.d.x1[14],Rv.d.x1[15],Rv.d.x1[16],Rv.d.x1[17],Rv.d.x1[18],Rv.d.x1[19],Rv.d.x1[20],Rv.d.x1[21],Rv.d.x1[22],Rv.d.x1.2[-(length(Rv.d.x1.2)-11):-length(Rv.d.x1.2)])
  
  # (t-13d) daily time point series data
  Rv.d.x1.2.13 <- c(Rv.d.x1[10],Rv.d.x1[11],Rv.d.x1[12],Rv.d.x1[13],Rv.d.x1[14],Rv.d.x1[15],Rv.d.x1[16],Rv.d.x1[17],Rv.d.x1[18],Rv.d.x1[19],Rv.d.x1[20],Rv.d.x1[21],Rv.d.x1[22],Rv.d.x1.2[-(length(Rv.d.x1.2)-12):-length(Rv.d.x1.2)])
  
  # (t-14d) daily time point series data
  Rv.d.x1.2.14 <- c(Rv.d.x1[9],Rv.d.x1[10],Rv.d.x1[11],Rv.d.x1[12],Rv.d.x1[13],Rv.d.x1[14],Rv.d.x1[15],Rv.d.x1[16],Rv.d.x1[17],Rv.d.x1[18],Rv.d.x1[19],Rv.d.x1[20],Rv.d.x1[21],Rv.d.x1[22],Rv.d.x1.2[-(length(Rv.d.x1.2)-13):-length(Rv.d.x1.2)])
  
  # (t-15d) daily time point series data
  Rv.d.x1.2.15 <- c(Rv.d.x1[8],Rv.d.x1[9],Rv.d.x1[10],Rv.d.x1[11],Rv.d.x1[12],Rv.d.x1[13],Rv.d.x1[14],Rv.d.x1[15],Rv.d.x1[16],Rv.d.x1[17],Rv.d.x1[18],Rv.d.x1[19],Rv.d.x1[20],Rv.d.x1[21],Rv.d.x1[22],Rv.d.x1.2[-(length(Rv.d.x1.2)-14):-length(Rv.d.x1.2)])
  
  # (t-16d) daily time point series data
  Rv.d.x1.2.16 <- c(Rv.d.x1[7],Rv.d.x1[8],Rv.d.x1[9],Rv.d.x1[10],Rv.d.x1[11],Rv.d.x1[12],Rv.d.x1[13],Rv.d.x1[14],Rv.d.x1[15],Rv.d.x1[16],Rv.d.x1[17],Rv.d.x1[18],Rv.d.x1[19],Rv.d.x1[20],Rv.d.x1[21],Rv.d.x1[22],Rv.d.x1.2[-(length(Rv.d.x1.2)-15):-length(Rv.d.x1.2)])
  
  # (t-17d) daily time point series data
  Rv.d.x1.2.17 <- c(Rv.d.x1[6],Rv.d.x1[7],Rv.d.x1[8],Rv.d.x1[9],Rv.d.x1[10],Rv.d.x1[11],Rv.d.x1[12],Rv.d.x1[13],Rv.d.x1[14],Rv.d.x1[15],Rv.d.x1[16],Rv.d.x1[17],Rv.d.x1[18],Rv.d.x1[19],Rv.d.x1[20],Rv.d.x1[21],Rv.d.x1[22],Rv.d.x1.2[-(length(Rv.d.x1.2)-16):-length(Rv.d.x1.2)])
  
  # (t-18d) daily time point series data
  Rv.d.x1.2.18 <- c(Rv.d.x1[5],Rv.d.x1[6],Rv.d.x1[7],Rv.d.x1[8],Rv.d.x1[9],Rv.d.x1[10],Rv.d.x1[11],Rv.d.x1[12],Rv.d.x1[13],Rv.d.x1[14],Rv.d.x1[15],Rv.d.x1[16],Rv.d.x1[17],Rv.d.x1[18],Rv.d.x1[19],Rv.d.x1[20],Rv.d.x1[21],Rv.d.x1[22],Rv.d.x1.2[-(length(Rv.d.x1.2)-17):-length(Rv.d.x1.2)])
  
  # (t-19d) daily time point series data
  Rv.d.x1.2.19 <- c(Rv.d.x1[4],Rv.d.x1[5],Rv.d.x1[6],Rv.d.x1[7],Rv.d.x1[8],Rv.d.x1[9],Rv.d.x1[10],Rv.d.x1[11],Rv.d.x1[12],Rv.d.x1[13],Rv.d.x1[14],Rv.d.x1[15],Rv.d.x1[16],Rv.d.x1[17],Rv.d.x1[18],Rv.d.x1[19],Rv.d.x1[20],Rv.d.x1[21],Rv.d.x1[22],Rv.d.x1.2[-(length(Rv.d.x1.2)-18):-length(Rv.d.x1.2)])
  
  # (t-20d) daily time point series data
  Rv.d.x1.2.20 <- c(Rv.d.x1[3],Rv.d.x1[4],Rv.d.x1[5],Rv.d.x1[6],Rv.d.x1[7],Rv.d.x1[8],Rv.d.x1[9],Rv.d.x1[10],Rv.d.x1[11],Rv.d.x1[12],Rv.d.x1[13],Rv.d.x1[14],Rv.d.x1[15],Rv.d.x1[16],Rv.d.x1[17],Rv.d.x1[18],Rv.d.x1[19],Rv.d.x1[20],Rv.d.x1[21],Rv.d.x1[22],Rv.d.x1.2[-(length(Rv.d.x1.2)-19):-length(Rv.d.x1.2)])
  
  # (t-21d) daily time point series data
  Rv.d.x1.2.21 <- c(Rv.d.x1[2],Rv.d.x1[3],Rv.d.x1[4],Rv.d.x1[5],Rv.d.x1[6],Rv.d.x1[7],Rv.d.x1[8],Rv.d.x1[9],Rv.d.x1[10],Rv.d.x1[11],Rv.d.x1[12],Rv.d.x1[13],Rv.d.x1[14],Rv.d.x1[15],Rv.d.x1[16],Rv.d.x1[17],Rv.d.x1[18],Rv.d.x1[19],Rv.d.x1[20],Rv.d.x1[21],Rv.d.x1[22],Rv.d.x1.2[-(length(Rv.d.x1.2)-20):-length(Rv.d.x1.2)])
  
  
  
  ## initial residual
  
  initial.res <-(Rv.d.y2-initial.AR22.model.coeffi[1]*Rv.d.x1.2-initial.AR22.model.coeffi[2]*Rv.d.x1.2.1-initial.AR22.model.coeffi[3]*Rv.d.x1.2.2
                 -initial.AR22.model.coeffi[4]*Rv.d.x1.2.3-initial.AR22.model.coeffi[5]*Rv.d.x1.2.4-initial.AR22.model.coeffi[6]*Rv.d.x1.2.5-initial.AR22.model.coeffi[7]*Rv.d.x1.2.6
                 -initial.AR22.model.coeffi[8]*Rv.d.x1.2.7-initial.AR22.model.coeffi[9]*Rv.d.x1.2.8-initial.AR22.model.coeffi[10]*Rv.d.x1.2.9-initial.AR22.model.coeffi[11]*Rv.d.x1.2.10-initial.AR22.model.coeffi[12]*Rv.d.x1.2.11
                 -initial.AR22.model.coeffi[13]*Rv.d.x1.2.12-initial.AR22.model.coeffi[14]*Rv.d.x1.2.13-initial.AR22.model.coeffi[15]*Rv.d.x1.2.14-initial.AR22.model.coeffi[16]*Rv.d.x1.2.15
                 -initial.AR22.model.coeffi[17]*Rv.d.x1.2.16-initial.AR22.model.coeffi[18]*Rv.d.x1.2.17-initial.AR22.model.coeffi[19]*Rv.d.x1.2.18-initial.AR22.model.coeffi[20]*Rv.d.x1.2.19-initial.AR22.model.coeffi[21]*Rv.d.x1.2.20
                 -initial.AR22.model.coeffi[22]*Rv.d.x1.2.21)
  
  
  RVcombined.data <- cbind(initial.res,Rv.d.x1.2,Rv.d.x1.2.1,Rv.d.x1.2.2,Rv.d.x1.2.3,
                           Rv.d.x1.2.4,Rv.d.x1.2.5,Rv.d.x1.2.6,Rv.d.x1.2.7,Rv.d.x1.2.8
                           ,Rv.d.x1.2.9,Rv.d.x1.2.10,Rv.d.x1.2.11,Rv.d.x1.2.12,Rv.d.x1.2.13
                           ,Rv.d.x1.2.14,Rv.d.x1.2.15,Rv.d.x1.2.16,Rv.d.x1.2.17,Rv.d.x1.2.18
                           ,Rv.d.x1.2.19,Rv.d.x1.2.20,Rv.d.x1.2.21)
  
########### STEP 2 : Initial NN fitting to initial residual set. 
  
  ## NN model fitting to initial res data.
  
  NN.model.fitting.to.initial.res <- neuralnet(initial.res~Rv.d.x1.2+Rv.d.x1.2.1+Rv.d.x1.2.2+Rv.d.x1.2.3+
                                                 Rv.d.x1.2.4+Rv.d.x1.2.5+Rv.d.x1.2.6+Rv.d.x1.2.7+Rv.d.x1.2.8
                                               +Rv.d.x1.2.9+Rv.d.x1.2.10+Rv.d.x1.2.11+Rv.d.x1.2.12+Rv.d.x1.2.13
                                               +Rv.d.x1.2.14+Rv.d.x1.2.15+Rv.d.x1.2.16+Rv.d.x1.2.17+Rv.d.x1.2.18
                                               +Rv.d.x1.2.19+Rv.d.x1.2.20+Rv.d.x1.2.21,data=RVcombined.data,
                                               hidden=10,linear.output=T,act.fct = "tanh")
  
  
  ## The AR(22) model coefficient value corresponding to RV_[t] (numeric vector to store the updated value)
  AR22.model.coeffi.beta01.vector <- numeric(r)
  u.AR22.model.coeffi.beta01.vector <- numeric(r)
  
  ## The AR(22) model coefficient value corresponding to RV_[t-1d]
  AR22.model.coeffi.beta02.vector <- numeric(r)
  u.AR22.model.coeffi.beta02.vector <- numeric(r)
  
  ## The same to RV_[t-2d]
  AR22.model.coeffi.beta03.vector <- numeric(r)
  u.AR22.model.coeffi.beta03.vector <- numeric(r)
  
  ## The same to RV_[t-3d]
  AR22.model.coeffi.beta04.vector <- numeric(r)
  u.AR22.model.coeffi.beta04.vector <- numeric(r)
  
  ## The same to RV_[t-4d]
  AR22.model.coeffi.beta05.vector <- numeric(r)
  u.AR22.model.coeffi.beta05.vector <- numeric(r)
  
  ## The same to RV_[t-5d]
  AR22.model.coeffi.beta06.vector <- numeric(r)
  u.AR22.model.coeffi.beta06.vector <- numeric(r)
  
  ## The same to RV_[t-6d]
  AR22.model.coeffi.beta07.vector <- numeric(r)
  u.AR22.model.coeffi.beta07.vector <- numeric(r)
  
  ## The same to RV_[t-7d]
  AR22.model.coeffi.beta08.vector <- numeric(r)
  u.AR22.model.coeffi.beta08.vector <- numeric(r)
  
  ## The same to RV_[t-8d]
  AR22.model.coeffi.beta09.vector <- numeric(r)
  u.AR22.model.coeffi.beta09.vector <- numeric(r)
  
  ## The same to RV_[t-9d]
  AR22.model.coeffi.beta010.vector <- numeric(r)
  u.AR22.model.coeffi.beta010.vector <- numeric(r)
  
  ## The same to RV_[t-10d]
  AR22.model.coeffi.beta011.vector <- numeric(r)
  u.AR22.model.coeffi.beta011.vector <- numeric(r)
  
  ## The same to RV_[t-11d]
  AR22.model.coeffi.beta012.vector <- numeric(r)
  u.AR22.model.coeffi.beta012.vector <- numeric(r)
  
  ## The same to RV_[t-12d]
  AR22.model.coeffi.beta013.vector <- numeric(r)
  u.AR22.model.coeffi.beta013.vector <- numeric(r)
  
  ## The same to RV_[t-13d]
  AR22.model.coeffi.beta014.vector <- numeric(r)
  u.AR22.model.coeffi.beta014.vector <- numeric(r)
  
  ## The same to RV_[t-14d]
  AR22.model.coeffi.beta015.vector <- numeric(r)
  u.AR22.model.coeffi.beta015.vector <- numeric(r)
  
  ## The same to RV_[t-15d]
  AR22.model.coeffi.beta016.vector <- numeric(r)
  u.AR22.model.coeffi.beta016.vector <- numeric(r)
  
  ## The same to RV_[t-16d]
  AR22.model.coeffi.beta017.vector <- numeric(r)
  u.AR22.model.coeffi.beta017.vector <- numeric(r)
  
  ## The same to RV_[t-17d]
  AR22.model.coeffi.beta018.vector <- numeric(r)
  u.AR22.model.coeffi.beta018.vector <- numeric(r)
  
  ## The same to RV_[t-18d]
  AR22.model.coeffi.beta019.vector <- numeric(r)
  u.AR22.model.coeffi.beta019.vector <- numeric(r)
  
  ## The same to RV_[t-19d]
  AR22.model.coeffi.beta020.vector <- numeric(r)
  u.AR22.model.coeffi.beta020.vector <- numeric(r)
  
  ## The same to RV_[t-20d]
  AR22.model.coeffi.beta021.vector <- numeric(r)
  u.AR22.model.coeffi.beta021.vector <- numeric(r)
  
  ## The same to RV_[t-21d]
  AR22.model.coeffi.beta022.vector <- numeric(r)
  u.AR22.model.coeffi.beta022.vector <- numeric(r)
  
  
  
  
  for(p in 1:r){
    
    
########### STEP 3 : refit.AR22.model to ( Rv_[t+1d](d) - fitted.value of 'NN.model.fitting.to.initial.res' )       
    
    refit.AR22.model <- arima((Rv.d.y2 - NN.model.fitting.to.initial.res$net.result[[1]]),order=c(22,0,0),method="CSS")
    
    ### coefficient
    
    refit.AR22.model.coeffi <-(refit.AR22.model$coef)
    refit.AR22.model.coeffi <- as.vector(refit.AR22.model.coeffi)
    refit.AR22.model.coeffi ## 
    
    
    ## store AR(22) Coefficients obtained from STEP 3
    
    
    AR22.model.coeffi.beta01.vector[p] <- refit.AR22.model.coeffi[1]
    
    AR22.model.coeffi.beta02.vector[p] <- refit.AR22.model.coeffi[2]
    
    AR22.model.coeffi.beta03.vector[p] <- refit.AR22.model.coeffi[3]
    
    AR22.model.coeffi.beta04.vector[p] <- refit.AR22.model.coeffi[4]
    
    AR22.model.coeffi.beta05.vector[p] <- refit.AR22.model.coeffi[5]
    
    AR22.model.coeffi.beta06.vector[p] <- refit.AR22.model.coeffi[6]
    
    AR22.model.coeffi.beta07.vector[p] <- refit.AR22.model.coeffi[7]
    
    AR22.model.coeffi.beta08.vector[p] <- refit.AR22.model.coeffi[8]
    
    AR22.model.coeffi.beta09.vector[p] <- refit.AR22.model.coeffi[9]
    
    AR22.model.coeffi.beta010.vector[p] <- refit.AR22.model.coeffi[10]
    
    AR22.model.coeffi.beta011.vector[p] <- refit.AR22.model.coeffi[11]
    
    AR22.model.coeffi.beta012.vector[p] <- refit.AR22.model.coeffi[12]
    
    AR22.model.coeffi.beta013.vector[p] <- refit.AR22.model.coeffi[13]
    
    AR22.model.coeffi.beta014.vector[p] <- refit.AR22.model.coeffi[14]
    
    AR22.model.coeffi.beta015.vector[p] <- refit.AR22.model.coeffi[15]
    
    AR22.model.coeffi.beta016.vector[p] <- refit.AR22.model.coeffi[16]
    
    AR22.model.coeffi.beta017.vector[p] <- refit.AR22.model.coeffi[17]
    
    AR22.model.coeffi.beta018.vector[p] <- refit.AR22.model.coeffi[18]
    
    AR22.model.coeffi.beta019.vector[p] <- refit.AR22.model.coeffi[19]
    
    AR22.model.coeffi.beta020.vector[p] <- refit.AR22.model.coeffi[20]
    
    AR22.model.coeffi.beta021.vector[p] <- refit.AR22.model.coeffi[21]
    
    AR22.model.coeffi.beta022.vector[p] <- refit.AR22.model.coeffi[22]
    
    
########### STEP 4 : Fit NN again to second.res data.
    
    
    # second.res
    second.res <-(Rv.d.y2-AR22.model.coeffi.beta01.vector[p]*Rv.d.x1.2-AR22.model.coeffi.beta02.vector[p]*Rv.d.x1.2.1-AR22.model.coeffi.beta03.vector[p]*Rv.d.x1.2.2-AR22.model.coeffi.beta04.vector[p]*Rv.d.x1.2.3-AR22.model.coeffi.beta05.vector[p]*Rv.d.x1.2.4-AR22.model.coeffi.beta06.vector[p]*Rv.d.x1.2.5-AR22.model.coeffi.beta07.vector[p]*Rv.d.x1.2.6-AR22.model.coeffi.beta08.vector[p]*Rv.d.x1.2.7
                  -AR22.model.coeffi.beta09.vector[p]*Rv.d.x1.2.8-AR22.model.coeffi.beta010.vector[p]*Rv.d.x1.2.9-AR22.model.coeffi.beta011.vector[p]*Rv.d.x1.2.10-AR22.model.coeffi.beta012.vector[p]*Rv.d.x1.2.11
                  -AR22.model.coeffi.beta013.vector[p]*Rv.d.x1.2.12-AR22.model.coeffi.beta014.vector[p]*Rv.d.x1.2.13-AR22.model.coeffi.beta015.vector[p]*Rv.d.x1.2.14-AR22.model.coeffi.beta016.vector[p]*Rv.d.x1.2.15
                  -AR22.model.coeffi.beta017.vector[p]*Rv.d.x1.2.16-AR22.model.coeffi.beta018.vector[p]*Rv.d.x1.2.17-AR22.model.coeffi.beta019.vector[p]*Rv.d.x1.2.18-AR22.model.coeffi.beta020.vector[p]*Rv.d.x1.2.19-AR22.model.coeffi.beta021.vector[p]*Rv.d.x1.2.20
                  -AR22.model.coeffi.beta022.vector[p]*Rv.d.x1.2.21)
    
    
    RVcombined.data2 <- cbind(second.res,Rv.d.x1.2,Rv.d.x1.2.1,Rv.d.x1.2.2,Rv.d.x1.2.3,
                              Rv.d.x1.2.4,Rv.d.x1.2.5,Rv.d.x1.2.6,Rv.d.x1.2.7,Rv.d.x1.2.8
                              ,Rv.d.x1.2.9,Rv.d.x1.2.10,Rv.d.x1.2.11,Rv.d.x1.2.12,Rv.d.x1.2.13
                              ,Rv.d.x1.2.14,Rv.d.x1.2.15,Rv.d.x1.2.16,Rv.d.x1.2.17,Rv.d.x1.2.18
                              ,Rv.d.x1.2.19,Rv.d.x1.2.20,Rv.d.x1.2.21)
    
    
## re-fit NN model to second.res data.
    refit.NN.model.to.second.res <- neuralnet(second.res~Rv.d.x1.2+Rv.d.x1.2.1+Rv.d.x1.2.2+Rv.d.x1.2.3+
                                                Rv.d.x1.2.4+Rv.d.x1.2.5+Rv.d.x1.2.6+Rv.d.x1.2.7+Rv.d.x1.2.8
                                              +Rv.d.x1.2.9+Rv.d.x1.2.10+Rv.d.x1.2.11+Rv.d.x1.2.12+Rv.d.x1.2.13
                                              +Rv.d.x1.2.14+Rv.d.x1.2.15+Rv.d.x1.2.16+Rv.d.x1.2.17+Rv.d.x1.2.18
                                              +Rv.d.x1.2.19+Rv.d.x1.2.20+Rv.d.x1.2.21,data=RVcombined.data2,
                                              hidden=10,linear.output=T,act.fct = "tanh")
    
    
    
############ Return to the STEP 3 : rerefit.AR22.model to ( Rv_[t+1d](d) - fitted.value of 'refit.NN.model.to.second.res' ) 
    
    rere.AR22.model.fit <- arima((Rv.d.y2 - refit.NN.model.to.second.res$net.result[[1]]),order=c(22,0,0),method="CSS")
    
    ### coefficient
    
    rere.AR22.model.coeffi <-(rere.AR22.model.fit$coef)
    rere.AR22.model.coeffi <- as.vector(rere.AR22.model.coeffi)
    
    
    
    
    ## store updated AR(22) Coefficients obtained from 'Return to the STEP 3'.
    
    u.AR22.model.coeffi.beta01.vector[p] <- rere.AR22.model.coeffi[1] 
    
    u.AR22.model.coeffi.beta02.vector[p] <- rere.AR22.model.coeffi[2] 
    
    u.AR22.model.coeffi.beta03.vector[p] <- rere.AR22.model.coeffi[3] 
    
    u.AR22.model.coeffi.beta04.vector[p] <- rere.AR22.model.coeffi[4] 
    
    u.AR22.model.coeffi.beta05.vector[p] <- rere.AR22.model.coeffi[5] 
    
    u.AR22.model.coeffi.beta06.vector[p] <- rere.AR22.model.coeffi[6] 
    
    u.AR22.model.coeffi.beta07.vector[p] <- rere.AR22.model.coeffi[7] 
    
    u.AR22.model.coeffi.beta08.vector[p] <- rere.AR22.model.coeffi[8] 
    
    u.AR22.model.coeffi.beta09.vector[p] <- rere.AR22.model.coeffi[9] 
    
    u.AR22.model.coeffi.beta010.vector[p] <- rere.AR22.model.coeffi[10] 
    
    u.AR22.model.coeffi.beta011.vector[p] <- rere.AR22.model.coeffi[11] 
    
    u.AR22.model.coeffi.beta012.vector[p] <- rere.AR22.model.coeffi[12] 
    
    u.AR22.model.coeffi.beta013.vector[p] <- rere.AR22.model.coeffi[13] 
    
    u.AR22.model.coeffi.beta014.vector[p] <- rere.AR22.model.coeffi[14] 
    
    u.AR22.model.coeffi.beta015.vector[p] <- rere.AR22.model.coeffi[15] 
    
    u.AR22.model.coeffi.beta016.vector[p] <- rere.AR22.model.coeffi[16] 
    
    u.AR22.model.coeffi.beta017.vector[p] <- rere.AR22.model.coeffi[17] 
    
    u.AR22.model.coeffi.beta018.vector[p] <- rere.AR22.model.coeffi[18] 
    
    u.AR22.model.coeffi.beta019.vector[p] <- rere.AR22.model.coeffi[19] 
    
    u.AR22.model.coeffi.beta020.vector[p] <- rere.AR22.model.coeffi[20] 
    
    u.AR22.model.coeffi.beta021.vector[p] <- rere.AR22.model.coeffi[21] 
    
    u.AR22.model.coeffi.beta022.vector[p] <- rere.AR22.model.coeffi[22] 
    
    
    ## 
    
    diff1=abs(AR22.model.coeffi.beta01.vector[p]-u.AR22.model.coeffi.beta01.vector[p])
    diff2=abs(AR22.model.coeffi.beta02.vector[p]-u.AR22.model.coeffi.beta02.vector[p])
    diff3=abs(AR22.model.coeffi.beta03.vector[p]-u.AR22.model.coeffi.beta03.vector[p])
    diff4=abs(AR22.model.coeffi.beta04.vector[p]-u.AR22.model.coeffi.beta04.vector[p])
    diff5=abs(AR22.model.coeffi.beta05.vector[p]-u.AR22.model.coeffi.beta05.vector[p])
    diff6=abs(AR22.model.coeffi.beta06.vector[p]-u.AR22.model.coeffi.beta06.vector[p])
    diff7=abs(AR22.model.coeffi.beta07.vector[p]-u.AR22.model.coeffi.beta07.vector[p])
    diff8=abs(AR22.model.coeffi.beta08.vector[p]-u.AR22.model.coeffi.beta08.vector[p])
    diff9=abs(AR22.model.coeffi.beta09.vector[p]-u.AR22.model.coeffi.beta09.vector[p])
    diff10=abs(AR22.model.coeffi.beta010.vector[p]-u.AR22.model.coeffi.beta010.vector[p])
    diff11=abs(AR22.model.coeffi.beta011.vector[p]-u.AR22.model.coeffi.beta011.vector[p])
    diff12=abs(AR22.model.coeffi.beta012.vector[p]-u.AR22.model.coeffi.beta012.vector[p])
    diff13=abs(AR22.model.coeffi.beta013.vector[p]-u.AR22.model.coeffi.beta013.vector[p])
    diff14=abs(AR22.model.coeffi.beta014.vector[p]-u.AR22.model.coeffi.beta014.vector[p])
    diff15=abs(AR22.model.coeffi.beta015.vector[p]-u.AR22.model.coeffi.beta015.vector[p])
    diff16=abs(AR22.model.coeffi.beta016.vector[p]-u.AR22.model.coeffi.beta016.vector[p])
    diff17=abs(AR22.model.coeffi.beta017.vector[p]-u.AR22.model.coeffi.beta017.vector[p])
    diff18=abs(AR22.model.coeffi.beta018.vector[p]-u.AR22.model.coeffi.beta018.vector[p])
    diff19=abs(AR22.model.coeffi.beta019.vector[p]-u.AR22.model.coeffi.beta019.vector[p])
    diff20=abs(AR22.model.coeffi.beta020.vector[p]-u.AR22.model.coeffi.beta020.vector[p])
    diff21=abs(AR22.model.coeffi.beta021.vector[p]-u.AR22.model.coeffi.beta021.vector[p])
    diff22=abs(AR22.model.coeffi.beta022.vector[p]-u.AR22.model.coeffi.beta022.vector[p])
    
    
    
    
    if( sum(c(diff1,diff2,diff3,diff4,diff5,diff6,diff7,diff8,diff9,diff10,diff11,diff12,diff13,diff14,diff15,diff16,diff17,diff18,diff19,diff20,diff21,diff22)<0.4)==22 )
    {
      
      # convergent coefficient corresponding to RV_[t] in AR(22) model part of HAR-infty-NN model. 
      AR22.model.t.coeffi <- as.numeric(AR22.model.coeffi.beta01.vector[p])
      
      # The same to RV_[t-1d]
      AR22.model.t.1.coeffi <- as.numeric(AR22.model.coeffi.beta02.vector[p])
      
      # The same to RV_[t-2d]
      AR22.model.t.2.coeffi <- as.numeric(AR22.model.coeffi.beta03.vector[p])
      
      # The same to RV_[t-3d]
      AR22.model.t.3.coeffi <- as.numeric(AR22.model.coeffi.beta04.vector[p])
      
      # The same to RV_[t-4d]
      AR22.model.t.4.coeffi <- as.numeric(AR22.model.coeffi.beta05.vector[p])
      
      # The same to RV_[t-5d]
      AR22.model.t.5.coeffi <- as.numeric(AR22.model.coeffi.beta06.vector[p])
      
      # The same to RV_[t-6d]
      AR22.model.t.6.coeffi <- as.numeric(AR22.model.coeffi.beta07.vector[p])
      
      # The same to RV_[t-7d]
      AR22.model.t.7.coeffi <- as.numeric(AR22.model.coeffi.beta08.vector[p])
      
      # The same to RV_[t-8d]
      AR22.model.t.8.coeffi <- as.numeric(AR22.model.coeffi.beta09.vector[p])
      
      # The same to RV_[t-9d]
      AR22.model.t.9.coeffi <- as.numeric(AR22.model.coeffi.beta010.vector[p])
      
      # The same to RV_[t-10d]
      AR22.model.t.10.coeffi <- as.numeric(AR22.model.coeffi.beta011.vector[p])
      
      # The same to RV_[t-11d]
      AR22.model.t.11.coeffi <- as.numeric(AR22.model.coeffi.beta012.vector[p])
      
      # The same to RV_[t-12d]
      AR22.model.t.12.coeffi <- as.numeric(AR22.model.coeffi.beta013.vector[p])
      
      # The same to RV_[t-13d]
      AR22.model.t.13.coeffi <- as.numeric(AR22.model.coeffi.beta014.vector[p])
      
      # The same to RV_[t-14d]
      AR22.model.t.14.coeffi <- as.numeric(AR22.model.coeffi.beta015.vector[p])
      
      # The same to RV_[t-15d]
      AR22.model.t.15.coeffi <- as.numeric(AR22.model.coeffi.beta016.vector[p])
      
      # The same to RV_[t-16d]
      AR22.model.t.16.coeffi <- as.numeric(AR22.model.coeffi.beta017.vector[p])
      
      # The same to RV_[t-17d]
      AR22.model.t.17.coeffi <- as.numeric(AR22.model.coeffi.beta018.vector[p])
      
      # The same to RV_[t-18d]
      AR22.model.t.18.coeffi <- as.numeric(AR22.model.coeffi.beta019.vector[p])
      
      # The same to RV_[t-19d]
      AR22.model.t.19.coeffi <- as.numeric(AR22.model.coeffi.beta020.vector[p])
      
      # The same to RV_[t-20d]
      AR22.model.t.20.coeffi <- as.numeric(AR22.model.coeffi.beta021.vector[p])
      
      # The same to RV_[t-21d]
      AR22.model.t.21.coeffi <- as.numeric(AR22.model.coeffi.beta022.vector[p])
      
      
      break
    }
    
    ## Repeat the above steps (STEP3-> STEP4-> STEP3) repeatedly 
    ## until the coefficients corresponding to the AR(22) model PART of HAR-infty-NN model 
    ## converge to a certain level. 
    
    
  }
  
  
  
  
########### STEP 5 : Fit the NN model lastly to the final.res data. 
  
  final.res <-(Rv.d.y2-AR22.model.t.coeffi*Rv.d.x1.2-AR22.model.t.1.coeffi*Rv.d.x1.2.1-AR22.model.t.2.coeffi*Rv.d.x1.2.2-AR22.model.t.3.coeffi*Rv.d.x1.2.3-AR22.model.t.4.coeffi*Rv.d.x1.2.4-AR22.model.t.5.coeffi*Rv.d.x1.2.5-AR22.model.t.6.coeffi*Rv.d.x1.2.6-AR22.model.t.7.coeffi*Rv.d.x1.2.7
               -AR22.model.t.8.coeffi*Rv.d.x1.2.8-AR22.model.t.9.coeffi*Rv.d.x1.2.9-AR22.model.t.10.coeffi*Rv.d.x1.2.10-AR22.model.t.11.coeffi*Rv.d.x1.2.11
               -AR22.model.t.12.coeffi*Rv.d.x1.2.12-AR22.model.t.13.coeffi*Rv.d.x1.2.13-AR22.model.t.14.coeffi*Rv.d.x1.2.14-AR22.model.t.15.coeffi*Rv.d.x1.2.15
               -AR22.model.t.16.coeffi*Rv.d.x1.2.16-AR22.model.t.17.coeffi*Rv.d.x1.2.17-AR22.model.t.18.coeffi*Rv.d.x1.2.18-AR22.model.t.19.coeffi*Rv.d.x1.2.19-AR22.model.t.20.coeffi*Rv.d.x1.2.20
               -AR22.model.t.21.coeffi*Rv.d.x1.2.21)
  
  ## 
  RVcombined.data3 <- cbind(final.res,Rv.d.x1.2,Rv.d.x1.2.1,Rv.d.x1.2.2,Rv.d.x1.2.3,
                            Rv.d.x1.2.4,Rv.d.x1.2.5,Rv.d.x1.2.6,Rv.d.x1.2.7,Rv.d.x1.2.8
                            ,Rv.d.x1.2.9,Rv.d.x1.2.10,Rv.d.x1.2.11,Rv.d.x1.2.12,Rv.d.x1.2.13
                            ,Rv.d.x1.2.14,Rv.d.x1.2.15,Rv.d.x1.2.16,Rv.d.x1.2.17,Rv.d.x1.2.18
                            ,Rv.d.x1.2.19,Rv.d.x1.2.20,Rv.d.x1.2.21)
  
  
  
  RVcombined.data3 <-as.data.frame(RVcombined.data3)
  
  
  
  NN.model.refitting.to.final.res <- neuralnet(final.res~Rv.d.x1.2+Rv.d.x1.2.1+Rv.d.x1.2.2+Rv.d.x1.2.3+
                                                 Rv.d.x1.2.4+Rv.d.x1.2.5+Rv.d.x1.2.6+Rv.d.x1.2.7+Rv.d.x1.2.8
                                               +Rv.d.x1.2.9+Rv.d.x1.2.10+Rv.d.x1.2.11+Rv.d.x1.2.12+Rv.d.x1.2.13
                                               +Rv.d.x1.2.14+Rv.d.x1.2.15+Rv.d.x1.2.16+Rv.d.x1.2.17+Rv.d.x1.2.18
                                               +Rv.d.x1.2.19+Rv.d.x1.2.20+Rv.d.x1.2.21,data=RVcombined.data3,
                                               hidden=10,linear.output=T,act.fct = "tanh")
  
################ tanh function values.
  
  a1 <- tanh(NN.model.refitting.to.final.res$weights[[1]][[1]][,1][1] + NN.model.refitting.to.final.res$weights[[1]][[1]][,1][2]*Rv.d.x1[nrow(RV.data)-i] + NN.model.refitting.to.final.res$weights[[1]][[1]][,1][3]*Rv.d.x1[nrow(RV.data)-(i+1)]+
                      NN.model.refitting.to.final.res$weights[[1]][[1]][,1][4]*Rv.d.x1[nrow(RV.data)-(i+2)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,1][5]*Rv.d.x1[nrow(RV.data)-(i+3)]+
                      NN.model.refitting.to.final.res$weights[[1]][[1]][,1][6]*Rv.d.x1[nrow(RV.data)-(i+4)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,1][7]*Rv.d.x1[nrow(RV.data)-(i+5)]+
                      +NN.model.refitting.to.final.res$weights[[1]][[1]][,1][8]*Rv.d.x1[nrow(RV.data)-(i+6)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,1][9]*Rv.d.x1[nrow(RV.data)-(i+7)]+
                      NN.model.refitting.to.final.res$weights[[1]][[1]][,1][10]*Rv.d.x1[nrow(RV.data)-(i+8)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,1][11]*Rv.d.x1[nrow(RV.data)-(i+9)]+
                      NN.model.refitting.to.final.res$weights[[1]][[1]][,1][12]*Rv.d.x1[nrow(RV.data)-(i+10)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,1][13]*Rv.d.x1[nrow(RV.data)-(i+11)]+
                      NN.model.refitting.to.final.res$weights[[1]][[1]][,1][14]*Rv.d.x1[nrow(RV.data)-(i+12)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,1][15]*Rv.d.x1[nrow(RV.data)-(i+13)]+
                      NN.model.refitting.to.final.res$weights[[1]][[1]][,1][16]*Rv.d.x1[nrow(RV.data)-(i+14)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,1][17]*Rv.d.x1[nrow(RV.data)-(i+15)]+
                      NN.model.refitting.to.final.res$weights[[1]][[1]][,1][18]*Rv.d.x1[nrow(RV.data)-(i+16)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,1][19]*Rv.d.x1[nrow(RV.data)-(i+17)]+
                      NN.model.refitting.to.final.res$weights[[1]][[1]][,1][20]*Rv.d.x1[nrow(RV.data)-(i+18)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,1][21]*Rv.d.x1[nrow(RV.data)-(i+19)]+
                      NN.model.refitting.to.final.res$weights[[1]][[1]][,1][22]*Rv.d.x1[nrow(RV.data)-(i+20)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,1][23]*Rv.d.x1[nrow(RV.data)-(i+21)])
  
  
  a2 <- tanh(NN.model.refitting.to.final.res$weights[[1]][[1]][,2][1] + NN.model.refitting.to.final.res$weights[[1]][[1]][,2][2]*Rv.d.x1[nrow(RV.data)-i] + NN.model.refitting.to.final.res$weights[[1]][[1]][,2][3]*Rv.d.x1[nrow(RV.data)-(i+1)]+
                      NN.model.refitting.to.final.res$weights[[1]][[1]][,2][4]*Rv.d.x1[nrow(RV.data)-(i+2)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,2][5]*Rv.d.x1[nrow(RV.data)-(i+3)]+
                      NN.model.refitting.to.final.res$weights[[1]][[1]][,2][6]*Rv.d.x1[nrow(RV.data)-(i+4)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,2][7]*Rv.d.x1[nrow(RV.data)-(i+5)]+
                      +NN.model.refitting.to.final.res$weights[[1]][[1]][,2][8]*Rv.d.x1[nrow(RV.data)-(i+6)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,2][9]*Rv.d.x1[nrow(RV.data)-(i+7)]+
                      NN.model.refitting.to.final.res$weights[[1]][[1]][,2][10]*Rv.d.x1[nrow(RV.data)-(i+8)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,2][11]*Rv.d.x1[nrow(RV.data)-(i+9)]+
                      NN.model.refitting.to.final.res$weights[[1]][[1]][,2][12]*Rv.d.x1[nrow(RV.data)-(i+10)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,2][13]*Rv.d.x1[nrow(RV.data)-(i+11)]+
                      NN.model.refitting.to.final.res$weights[[1]][[1]][,2][14]*Rv.d.x1[nrow(RV.data)-(i+12)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,2][15]*Rv.d.x1[nrow(RV.data)-(i+13)]+
                      NN.model.refitting.to.final.res$weights[[1]][[1]][,2][16]*Rv.d.x1[nrow(RV.data)-(i+14)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,2][17]*Rv.d.x1[nrow(RV.data)-(i+15)]+
                      NN.model.refitting.to.final.res$weights[[1]][[1]][,2][18]*Rv.d.x1[nrow(RV.data)-(i+16)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,2][19]*Rv.d.x1[nrow(RV.data)-(i+17)]+
                      NN.model.refitting.to.final.res$weights[[1]][[1]][,2][20]*Rv.d.x1[nrow(RV.data)-(i+18)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,2][21]*Rv.d.x1[nrow(RV.data)-(i+19)]+
                      NN.model.refitting.to.final.res$weights[[1]][[1]][,2][22]*Rv.d.x1[nrow(RV.data)-(i+20)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,2][23]*Rv.d.x1[nrow(RV.data)-(i+21)])
  
  a3 <- tanh(NN.model.refitting.to.final.res$weights[[1]][[1]][,3][1] + NN.model.refitting.to.final.res$weights[[1]][[1]][,3][2]*Rv.d.x1[nrow(RV.data)-i] + NN.model.refitting.to.final.res$weights[[1]][[1]][,3][3]*Rv.d.x1[nrow(RV.data)-(i+1)]+
                      NN.model.refitting.to.final.res$weights[[1]][[1]][,3][4]*Rv.d.x1[nrow(RV.data)-(i+2)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,3][5]*Rv.d.x1[nrow(RV.data)-(i+3)]+
                      NN.model.refitting.to.final.res$weights[[1]][[1]][,3][6]*Rv.d.x1[nrow(RV.data)-(i+4)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,3][7]*Rv.d.x1[nrow(RV.data)-(i+5)]+
                      +NN.model.refitting.to.final.res$weights[[1]][[1]][,3][8]*Rv.d.x1[nrow(RV.data)-(i+6)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,3][9]*Rv.d.x1[nrow(RV.data)-(i+7)]+
                      NN.model.refitting.to.final.res$weights[[1]][[1]][,3][10]*Rv.d.x1[nrow(RV.data)-(i+8)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,3][11]*Rv.d.x1[nrow(RV.data)-(i+9)]+
                      NN.model.refitting.to.final.res$weights[[1]][[1]][,3][12]*Rv.d.x1[nrow(RV.data)-(i+10)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,3][13]*Rv.d.x1[nrow(RV.data)-(i+11)]+
                      NN.model.refitting.to.final.res$weights[[1]][[1]][,3][14]*Rv.d.x1[nrow(RV.data)-(i+12)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,3][15]*Rv.d.x1[nrow(RV.data)-(i+13)]+
                      NN.model.refitting.to.final.res$weights[[1]][[1]][,3][16]*Rv.d.x1[nrow(RV.data)-(i+14)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,3][17]*Rv.d.x1[nrow(RV.data)-(i+15)]+
                      NN.model.refitting.to.final.res$weights[[1]][[1]][,3][18]*Rv.d.x1[nrow(RV.data)-(i+16)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,3][19]*Rv.d.x1[nrow(RV.data)-(i+17)]+
                      NN.model.refitting.to.final.res$weights[[1]][[1]][,3][20]*Rv.d.x1[nrow(RV.data)-(i+18)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,3][21]*Rv.d.x1[nrow(RV.data)-(i+19)]+
                      NN.model.refitting.to.final.res$weights[[1]][[1]][,3][22]*Rv.d.x1[nrow(RV.data)-(i+20)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,3][23]*Rv.d.x1[nrow(RV.data)-(i+21)])
  
  a4 <- tanh(NN.model.refitting.to.final.res$weights[[1]][[1]][,4][1] + NN.model.refitting.to.final.res$weights[[1]][[1]][,4][2]*Rv.d.x1[nrow(RV.data)-i] + NN.model.refitting.to.final.res$weights[[1]][[1]][,4][3]*Rv.d.x1[nrow(RV.data)-(i+1)]+
                      NN.model.refitting.to.final.res$weights[[1]][[1]][,4][4]*Rv.d.x1[nrow(RV.data)-(i+2)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,4][5]*Rv.d.x1[nrow(RV.data)-(i+3)]+
                      NN.model.refitting.to.final.res$weights[[1]][[1]][,4][6]*Rv.d.x1[nrow(RV.data)-(i+4)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,4][7]*Rv.d.x1[nrow(RV.data)-(i+5)]+
                      +NN.model.refitting.to.final.res$weights[[1]][[1]][,4][8]*Rv.d.x1[nrow(RV.data)-(i+6)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,4][9]*Rv.d.x1[nrow(RV.data)-(i+7)]+
                      NN.model.refitting.to.final.res$weights[[1]][[1]][,4][10]*Rv.d.x1[nrow(RV.data)-(i+8)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,4][11]*Rv.d.x1[nrow(RV.data)-(i+9)]+
                      NN.model.refitting.to.final.res$weights[[1]][[1]][,4][12]*Rv.d.x1[nrow(RV.data)-(i+10)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,4][13]*Rv.d.x1[nrow(RV.data)-(i+11)]+
                      NN.model.refitting.to.final.res$weights[[1]][[1]][,4][14]*Rv.d.x1[nrow(RV.data)-(i+12)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,4][15]*Rv.d.x1[nrow(RV.data)-(i+13)]+
                      NN.model.refitting.to.final.res$weights[[1]][[1]][,4][16]*Rv.d.x1[nrow(RV.data)-(i+14)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,4][17]*Rv.d.x1[nrow(RV.data)-(i+15)]+
                      NN.model.refitting.to.final.res$weights[[1]][[1]][,4][18]*Rv.d.x1[nrow(RV.data)-(i+16)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,4][19]*Rv.d.x1[nrow(RV.data)-(i+17)]+
                      NN.model.refitting.to.final.res$weights[[1]][[1]][,4][20]*Rv.d.x1[nrow(RV.data)-(i+18)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,4][21]*Rv.d.x1[nrow(RV.data)-(i+19)]+
                      NN.model.refitting.to.final.res$weights[[1]][[1]][,4][22]*Rv.d.x1[nrow(RV.data)-(i+20)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,4][23]*Rv.d.x1[nrow(RV.data)-(i+21)])
  
  
  a5 <- tanh(NN.model.refitting.to.final.res$weights[[1]][[1]][,5][1] + NN.model.refitting.to.final.res$weights[[1]][[1]][,5][2]*Rv.d.x1[nrow(RV.data)-i] + NN.model.refitting.to.final.res$weights[[1]][[1]][,5][3]*Rv.d.x1[nrow(RV.data)-(i+1)]+
                      NN.model.refitting.to.final.res$weights[[1]][[1]][,5][4]*Rv.d.x1[nrow(RV.data)-(i+2)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,5][5]*Rv.d.x1[nrow(RV.data)-(i+3)]+
                      NN.model.refitting.to.final.res$weights[[1]][[1]][,5][6]*Rv.d.x1[nrow(RV.data)-(i+4)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,5][7]*Rv.d.x1[nrow(RV.data)-(i+5)]+
                      +NN.model.refitting.to.final.res$weights[[1]][[1]][,5][8]*Rv.d.x1[nrow(RV.data)-(i+6)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,5][9]*Rv.d.x1[nrow(RV.data)-(i+7)]+
                      NN.model.refitting.to.final.res$weights[[1]][[1]][,5][10]*Rv.d.x1[nrow(RV.data)-(i+8)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,5][11]*Rv.d.x1[nrow(RV.data)-(i+9)]+
                      NN.model.refitting.to.final.res$weights[[1]][[1]][,5][12]*Rv.d.x1[nrow(RV.data)-(i+10)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,5][13]*Rv.d.x1[nrow(RV.data)-(i+11)]+
                      NN.model.refitting.to.final.res$weights[[1]][[1]][,5][14]*Rv.d.x1[nrow(RV.data)-(i+12)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,5][15]*Rv.d.x1[nrow(RV.data)-(i+13)]+
                      NN.model.refitting.to.final.res$weights[[1]][[1]][,5][16]*Rv.d.x1[nrow(RV.data)-(i+14)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,5][17]*Rv.d.x1[nrow(RV.data)-(i+15)]+
                      NN.model.refitting.to.final.res$weights[[1]][[1]][,5][18]*Rv.d.x1[nrow(RV.data)-(i+16)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,5][19]*Rv.d.x1[nrow(RV.data)-(i+17)]+
                      NN.model.refitting.to.final.res$weights[[1]][[1]][,5][20]*Rv.d.x1[nrow(RV.data)-(i+18)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,5][21]*Rv.d.x1[nrow(RV.data)-(i+19)]+
                      NN.model.refitting.to.final.res$weights[[1]][[1]][,5][22]*Rv.d.x1[nrow(RV.data)-(i+20)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,5][23]*Rv.d.x1[nrow(RV.data)-(i+21)])
  
  
  a6 <- tanh(NN.model.refitting.to.final.res$weights[[1]][[1]][,6][1] + NN.model.refitting.to.final.res$weights[[1]][[1]][,6][2]*Rv.d.x1[nrow(RV.data)-i] + NN.model.refitting.to.final.res$weights[[1]][[1]][,6][3]*Rv.d.x1[nrow(RV.data)-(i+1)]+
               NN.model.refitting.to.final.res$weights[[1]][[1]][,6][4]*Rv.d.x1[nrow(RV.data)-(i+2)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,6][5]*Rv.d.x1[nrow(RV.data)-(i+3)]+
               NN.model.refitting.to.final.res$weights[[1]][[1]][,6][6]*Rv.d.x1[nrow(RV.data)-(i+4)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,6][7]*Rv.d.x1[nrow(RV.data)-(i+5)]+
               +NN.model.refitting.to.final.res$weights[[1]][[1]][,6][8]*Rv.d.x1[nrow(RV.data)-(i+6)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,6][9]*Rv.d.x1[nrow(RV.data)-(i+7)]+
               NN.model.refitting.to.final.res$weights[[1]][[1]][,6][10]*Rv.d.x1[nrow(RV.data)-(i+8)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,6][11]*Rv.d.x1[nrow(RV.data)-(i+9)]+
               NN.model.refitting.to.final.res$weights[[1]][[1]][,6][12]*Rv.d.x1[nrow(RV.data)-(i+10)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,6][13]*Rv.d.x1[nrow(RV.data)-(i+11)]+
               NN.model.refitting.to.final.res$weights[[1]][[1]][,6][14]*Rv.d.x1[nrow(RV.data)-(i+12)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,6][15]*Rv.d.x1[nrow(RV.data)-(i+13)]+
               NN.model.refitting.to.final.res$weights[[1]][[1]][,6][16]*Rv.d.x1[nrow(RV.data)-(i+14)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,6][17]*Rv.d.x1[nrow(RV.data)-(i+15)]+
               NN.model.refitting.to.final.res$weights[[1]][[1]][,6][18]*Rv.d.x1[nrow(RV.data)-(i+16)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,6][19]*Rv.d.x1[nrow(RV.data)-(i+17)]+
               NN.model.refitting.to.final.res$weights[[1]][[1]][,6][20]*Rv.d.x1[nrow(RV.data)-(i+18)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,6][21]*Rv.d.x1[nrow(RV.data)-(i+19)]+
               NN.model.refitting.to.final.res$weights[[1]][[1]][,6][22]*Rv.d.x1[nrow(RV.data)-(i+20)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,6][23]*Rv.d.x1[nrow(RV.data)-(i+21)])
  
  
  a7 <- tanh(NN.model.refitting.to.final.res$weights[[1]][[1]][,7][1] + NN.model.refitting.to.final.res$weights[[1]][[1]][,7][2]*Rv.d.x1[nrow(RV.data)-i] + NN.model.refitting.to.final.res$weights[[1]][[1]][,7][3]*Rv.d.x1[nrow(RV.data)-(i+1)]+
               NN.model.refitting.to.final.res$weights[[1]][[1]][,7][4]*Rv.d.x1[nrow(RV.data)-(i+2)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,7][5]*Rv.d.x1[nrow(RV.data)-(i+3)]+
               NN.model.refitting.to.final.res$weights[[1]][[1]][,7][6]*Rv.d.x1[nrow(RV.data)-(i+4)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,7][7]*Rv.d.x1[nrow(RV.data)-(i+5)]+
               +NN.model.refitting.to.final.res$weights[[1]][[1]][,7][8]*Rv.d.x1[nrow(RV.data)-(i+6)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,7][9]*Rv.d.x1[nrow(RV.data)-(i+7)]+
               NN.model.refitting.to.final.res$weights[[1]][[1]][,7][10]*Rv.d.x1[nrow(RV.data)-(i+8)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,7][11]*Rv.d.x1[nrow(RV.data)-(i+9)]+
               NN.model.refitting.to.final.res$weights[[1]][[1]][,7][12]*Rv.d.x1[nrow(RV.data)-(i+10)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,7][13]*Rv.d.x1[nrow(RV.data)-(i+11)]+
               NN.model.refitting.to.final.res$weights[[1]][[1]][,7][14]*Rv.d.x1[nrow(RV.data)-(i+12)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,7][15]*Rv.d.x1[nrow(RV.data)-(i+13)]+
               NN.model.refitting.to.final.res$weights[[1]][[1]][,7][16]*Rv.d.x1[nrow(RV.data)-(i+14)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,7][17]*Rv.d.x1[nrow(RV.data)-(i+15)]+
               NN.model.refitting.to.final.res$weights[[1]][[1]][,7][18]*Rv.d.x1[nrow(RV.data)-(i+16)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,7][19]*Rv.d.x1[nrow(RV.data)-(i+17)]+
               NN.model.refitting.to.final.res$weights[[1]][[1]][,7][20]*Rv.d.x1[nrow(RV.data)-(i+18)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,7][21]*Rv.d.x1[nrow(RV.data)-(i+19)]+
               NN.model.refitting.to.final.res$weights[[1]][[1]][,7][22]*Rv.d.x1[nrow(RV.data)-(i+20)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,7][23]*Rv.d.x1[nrow(RV.data)-(i+21)])
  
  a8 <- tanh(NN.model.refitting.to.final.res$weights[[1]][[1]][,8][1] + NN.model.refitting.to.final.res$weights[[1]][[1]][,8][2]*Rv.d.x1[nrow(RV.data)-i] + NN.model.refitting.to.final.res$weights[[1]][[1]][,8][3]*Rv.d.x1[nrow(RV.data)-(i+1)]+
               NN.model.refitting.to.final.res$weights[[1]][[1]][,8][4]*Rv.d.x1[nrow(RV.data)-(i+2)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,8][5]*Rv.d.x1[nrow(RV.data)-(i+3)]+
               NN.model.refitting.to.final.res$weights[[1]][[1]][,8][6]*Rv.d.x1[nrow(RV.data)-(i+4)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,8][7]*Rv.d.x1[nrow(RV.data)-(i+5)]+
               +NN.model.refitting.to.final.res$weights[[1]][[1]][,8][8]*Rv.d.x1[nrow(RV.data)-(i+6)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,8][9]*Rv.d.x1[nrow(RV.data)-(i+7)]+
               NN.model.refitting.to.final.res$weights[[1]][[1]][,8][10]*Rv.d.x1[nrow(RV.data)-(i+8)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,8][11]*Rv.d.x1[nrow(RV.data)-(i+9)]+
               NN.model.refitting.to.final.res$weights[[1]][[1]][,8][12]*Rv.d.x1[nrow(RV.data)-(i+10)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,8][13]*Rv.d.x1[nrow(RV.data)-(i+11)]+
               NN.model.refitting.to.final.res$weights[[1]][[1]][,8][14]*Rv.d.x1[nrow(RV.data)-(i+12)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,8][15]*Rv.d.x1[nrow(RV.data)-(i+13)]+
               NN.model.refitting.to.final.res$weights[[1]][[1]][,8][16]*Rv.d.x1[nrow(RV.data)-(i+14)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,8][17]*Rv.d.x1[nrow(RV.data)-(i+15)]+
               NN.model.refitting.to.final.res$weights[[1]][[1]][,8][18]*Rv.d.x1[nrow(RV.data)-(i+16)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,8][19]*Rv.d.x1[nrow(RV.data)-(i+17)]+
               NN.model.refitting.to.final.res$weights[[1]][[1]][,8][20]*Rv.d.x1[nrow(RV.data)-(i+18)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,8][21]*Rv.d.x1[nrow(RV.data)-(i+19)]+
               NN.model.refitting.to.final.res$weights[[1]][[1]][,8][22]*Rv.d.x1[nrow(RV.data)-(i+20)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,8][23]*Rv.d.x1[nrow(RV.data)-(i+21)])
  
  a9 <- tanh(NN.model.refitting.to.final.res$weights[[1]][[1]][,9][1] + NN.model.refitting.to.final.res$weights[[1]][[1]][,9][2]*Rv.d.x1[nrow(RV.data)-i] + NN.model.refitting.to.final.res$weights[[1]][[1]][,9][3]*Rv.d.x1[nrow(RV.data)-(i+1)]+
               NN.model.refitting.to.final.res$weights[[1]][[1]][,9][4]*Rv.d.x1[nrow(RV.data)-(i+2)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,9][5]*Rv.d.x1[nrow(RV.data)-(i+3)]+
               NN.model.refitting.to.final.res$weights[[1]][[1]][,9][6]*Rv.d.x1[nrow(RV.data)-(i+4)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,9][7]*Rv.d.x1[nrow(RV.data)-(i+5)]+
               +NN.model.refitting.to.final.res$weights[[1]][[1]][,9][8]*Rv.d.x1[nrow(RV.data)-(i+6)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,9][9]*Rv.d.x1[nrow(RV.data)-(i+7)]+
               NN.model.refitting.to.final.res$weights[[1]][[1]][,9][10]*Rv.d.x1[nrow(RV.data)-(i+8)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,9][11]*Rv.d.x1[nrow(RV.data)-(i+9)]+
               NN.model.refitting.to.final.res$weights[[1]][[1]][,9][12]*Rv.d.x1[nrow(RV.data)-(i+10)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,9][13]*Rv.d.x1[nrow(RV.data)-(i+11)]+
               NN.model.refitting.to.final.res$weights[[1]][[1]][,9][14]*Rv.d.x1[nrow(RV.data)-(i+12)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,9][15]*Rv.d.x1[nrow(RV.data)-(i+13)]+
               NN.model.refitting.to.final.res$weights[[1]][[1]][,9][16]*Rv.d.x1[nrow(RV.data)-(i+14)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,9][17]*Rv.d.x1[nrow(RV.data)-(i+15)]+
               NN.model.refitting.to.final.res$weights[[1]][[1]][,9][18]*Rv.d.x1[nrow(RV.data)-(i+16)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,9][19]*Rv.d.x1[nrow(RV.data)-(i+17)]+
               NN.model.refitting.to.final.res$weights[[1]][[1]][,9][20]*Rv.d.x1[nrow(RV.data)-(i+18)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,9][21]*Rv.d.x1[nrow(RV.data)-(i+19)]+
               NN.model.refitting.to.final.res$weights[[1]][[1]][,9][22]*Rv.d.x1[nrow(RV.data)-(i+20)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,9][23]*Rv.d.x1[nrow(RV.data)-(i+21)])
  
  
  a10 <- tanh(NN.model.refitting.to.final.res$weights[[1]][[1]][,10][1] + NN.model.refitting.to.final.res$weights[[1]][[1]][,10][2]*Rv.d.x1[nrow(RV.data)-i] + NN.model.refitting.to.final.res$weights[[1]][[1]][,10][3]*Rv.d.x1[nrow(RV.data)-(i+1)]+
               NN.model.refitting.to.final.res$weights[[1]][[1]][,10][4]*Rv.d.x1[nrow(RV.data)-(i+2)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,10][5]*Rv.d.x1[nrow(RV.data)-(i+3)]+
               NN.model.refitting.to.final.res$weights[[1]][[1]][,10][6]*Rv.d.x1[nrow(RV.data)-(i+4)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,10][7]*Rv.d.x1[nrow(RV.data)-(i+5)]+
               +NN.model.refitting.to.final.res$weights[[1]][[1]][,10][8]*Rv.d.x1[nrow(RV.data)-(i+6)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,10][9]*Rv.d.x1[nrow(RV.data)-(i+7)]+
               NN.model.refitting.to.final.res$weights[[1]][[1]][,10][10]*Rv.d.x1[nrow(RV.data)-(i+8)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,10][11]*Rv.d.x1[nrow(RV.data)-(i+9)]+
               NN.model.refitting.to.final.res$weights[[1]][[1]][,10][12]*Rv.d.x1[nrow(RV.data)-(i+10)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,10][13]*Rv.d.x1[nrow(RV.data)-(i+11)]+
               NN.model.refitting.to.final.res$weights[[1]][[1]][,10][14]*Rv.d.x1[nrow(RV.data)-(i+12)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,10][15]*Rv.d.x1[nrow(RV.data)-(i+13)]+
               NN.model.refitting.to.final.res$weights[[1]][[1]][,10][16]*Rv.d.x1[nrow(RV.data)-(i+14)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,10][17]*Rv.d.x1[nrow(RV.data)-(i+15)]+
               NN.model.refitting.to.final.res$weights[[1]][[1]][,10][18]*Rv.d.x1[nrow(RV.data)-(i+16)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,10][19]*Rv.d.x1[nrow(RV.data)-(i+17)]+
               NN.model.refitting.to.final.res$weights[[1]][[1]][,10][20]*Rv.d.x1[nrow(RV.data)-(i+18)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,10][21]*Rv.d.x1[nrow(RV.data)-(i+19)]+
               NN.model.refitting.to.final.res$weights[[1]][[1]][,10][22]*Rv.d.x1[nrow(RV.data)-(i+20)]+NN.model.refitting.to.final.res$weights[[1]][[1]][,10][23]*Rv.d.x1[nrow(RV.data)-(i+21)])
  
  
  
### All of the coefficients in the HAR-infty-NN model.
  
  
  coefficient.list <- list("Beta_01" = AR22.model.t.coeffi, "Beta_02" = AR22.model.t.1.coeffi, "Beta_03" = AR22.model.t.2.coeffi,
                           "Beta_04" = AR22.model.t.3.coeffi, "Beta_05" = AR22.model.t.4.coeffi, "Beta_06" = AR22.model.t.5.coeffi,
                           "Beta_07" = AR22.model.t.6.coeffi, "Beta_08" = AR22.model.t.7.coeffi, "Beta_09" = AR22.model.t.8.coeffi,
                           "Beta_010" = AR22.model.t.9.coeffi, "Beta_011" = AR22.model.t.10.coeffi, "Beta_012" = AR22.model.t.11.coeffi,
                           "Beta_013" = AR22.model.t.12.coeffi, "Beta_014" = AR22.model.t.13.coeffi, "Beta_015" = AR22.model.t.14.coeffi,
                           "Beta_016" = AR22.model.t.15.coeffi, "Beta_017" = AR22.model.t.16.coeffi, "Beta_018" = AR22.model.t.17.coeffi,
                           "Beta_019" = AR22.model.t.18.coeffi, "Beta_020" = AR22.model.t.19.coeffi, "Beta_021" = AR22.model.t.20.coeffi,
                           "Beta_022" = AR22.model.t.21.coeffi, "Beta_00" = NN.model.refitting.to.final.res$weights[[1]][[2]][[1]], 
                           
                           "beta_1" = NN.model.refitting.to.final.res$weights[[1]][[2]][[2]], 
                           "gamma_10" = NN.model.refitting.to.final.res$weights[[1]][[1]][,1][1], "gamma_11" = NN.model.refitting.to.final.res$weights[[1]][[1]][,1][2], "gamma_12" = NN.model.refitting.to.final.res$weights[[1]][[1]][,1][3], 
                           "gamma_13" = NN.model.refitting.to.final.res$weights[[1]][[1]][,1][4], "gamma_14" = NN.model.refitting.to.final.res$weights[[1]][[1]][,1][5], "gamma_15" = NN.model.refitting.to.final.res$weights[[1]][[1]][,1][6],
                           "gamma_16" = NN.model.refitting.to.final.res$weights[[1]][[1]][,1][7], "gamma_17" = NN.model.refitting.to.final.res$weights[[1]][[1]][,1][8], "gamma_18" = NN.model.refitting.to.final.res$weights[[1]][[1]][,1][9],
                           "gamma_19" = NN.model.refitting.to.final.res$weights[[1]][[1]][,1][10], "gamma_110" = NN.model.refitting.to.final.res$weights[[1]][[1]][,1][11], "gamma_111" = NN.model.refitting.to.final.res$weights[[1]][[1]][,1][12],
                           "gamma_112" = NN.model.refitting.to.final.res$weights[[1]][[1]][,1][13], "gamma_113" = NN.model.refitting.to.final.res$weights[[1]][[1]][,1][14], "gamma_114" = NN.model.refitting.to.final.res$weights[[1]][[1]][,1][15],
                           "gamma_115" = NN.model.refitting.to.final.res$weights[[1]][[1]][,1][16], "gamma_116" = NN.model.refitting.to.final.res$weights[[1]][[1]][,1][17], "gamma_117" = NN.model.refitting.to.final.res$weights[[1]][[1]][,1][18],
                           "gamma_118" = NN.model.refitting.to.final.res$weights[[1]][[1]][,1][19], "gamma_119" = NN.model.refitting.to.final.res$weights[[1]][[1]][,1][20], "gamma_120" = NN.model.refitting.to.final.res$weights[[1]][[1]][,1][21], 
                           "gamma_121" = NN.model.refitting.to.final.res$weights[[1]][[1]][,1][22], "gamma_122" = NN.model.refitting.to.final.res$weights[[1]][[1]][,1][23],
                           
                           "beta_2" = NN.model.refitting.to.final.res$weights[[1]][[2]][[3]], 
                           "gamma_20" = NN.model.refitting.to.final.res$weights[[1]][[1]][,2][1], "gamma_21" = NN.model.refitting.to.final.res$weights[[1]][[1]][,2][2], "gamma_22" = NN.model.refitting.to.final.res$weights[[1]][[1]][,2][3], 
                           "gamma_23" = NN.model.refitting.to.final.res$weights[[1]][[1]][,2][4], "gamma_24" = NN.model.refitting.to.final.res$weights[[1]][[1]][,2][5], "gamma_25" = NN.model.refitting.to.final.res$weights[[1]][[1]][,2][6],
                           "gamma_26" = NN.model.refitting.to.final.res$weights[[1]][[1]][,2][7], "gamma_27" = NN.model.refitting.to.final.res$weights[[1]][[1]][,2][8], "gamma_28" = NN.model.refitting.to.final.res$weights[[1]][[1]][,2][9],
                           "gamma_29" = NN.model.refitting.to.final.res$weights[[1]][[1]][,2][10], "gamma_210" = NN.model.refitting.to.final.res$weights[[1]][[1]][,2][11], "gamma_211" = NN.model.refitting.to.final.res$weights[[1]][[1]][,2][12],
                           "gamma_212" = NN.model.refitting.to.final.res$weights[[1]][[1]][,2][13], "gamma_213" = NN.model.refitting.to.final.res$weights[[1]][[1]][,2][14], "gamma_214" = NN.model.refitting.to.final.res$weights[[1]][[1]][,2][15],
                           "gamma_215" = NN.model.refitting.to.final.res$weights[[1]][[1]][,2][16], "gamma_216" = NN.model.refitting.to.final.res$weights[[1]][[1]][,2][17], "gamma_217" = NN.model.refitting.to.final.res$weights[[1]][[1]][,2][18],
                           "gamma_218" = NN.model.refitting.to.final.res$weights[[1]][[1]][,2][19], "gamma_219" = NN.model.refitting.to.final.res$weights[[1]][[1]][,2][20], "gamma_220" = NN.model.refitting.to.final.res$weights[[1]][[1]][,2][21], 
                           "gamma_221" = NN.model.refitting.to.final.res$weights[[1]][[1]][,2][22], "gamma_222" = NN.model.refitting.to.final.res$weights[[1]][[1]][,2][23],
                           
                           "beta_3" = NN.model.refitting.to.final.res$weights[[1]][[2]][[4]], 
                           "gamma_30" = NN.model.refitting.to.final.res$weights[[1]][[1]][,3][1], "gamma_31" = NN.model.refitting.to.final.res$weights[[1]][[1]][,3][2], "gamma_32" = NN.model.refitting.to.final.res$weights[[1]][[1]][,3][3], 
                           "gamma_33" = NN.model.refitting.to.final.res$weights[[1]][[1]][,3][4], "gamma_34" = NN.model.refitting.to.final.res$weights[[1]][[1]][,3][5], "gamma_35" = NN.model.refitting.to.final.res$weights[[1]][[1]][,3][6],
                           "gamma_36" = NN.model.refitting.to.final.res$weights[[1]][[1]][,3][7], "gamma_37" = NN.model.refitting.to.final.res$weights[[1]][[1]][,3][8], "gamma_38" = NN.model.refitting.to.final.res$weights[[1]][[1]][,3][9],
                           "gamma_39" = NN.model.refitting.to.final.res$weights[[1]][[1]][,3][10], "gamma_310" = NN.model.refitting.to.final.res$weights[[1]][[1]][,3][11], "gamma_311" = NN.model.refitting.to.final.res$weights[[1]][[1]][,3][12],
                           "gamma_312" = NN.model.refitting.to.final.res$weights[[1]][[1]][,3][13], "gamma_313" = NN.model.refitting.to.final.res$weights[[1]][[1]][,3][14], "gamma_314" = NN.model.refitting.to.final.res$weights[[1]][[1]][,3][15],
                           "gamma_315" = NN.model.refitting.to.final.res$weights[[1]][[1]][,3][16], "gamma_316" = NN.model.refitting.to.final.res$weights[[1]][[1]][,3][17], "gamma_317" = NN.model.refitting.to.final.res$weights[[1]][[1]][,3][18],
                           "gamma_318" = NN.model.refitting.to.final.res$weights[[1]][[1]][,3][19], "gamma_319" = NN.model.refitting.to.final.res$weights[[1]][[1]][,3][20], "gamma_320" = NN.model.refitting.to.final.res$weights[[1]][[1]][,3][21], 
                           "gamma_321" = NN.model.refitting.to.final.res$weights[[1]][[1]][,3][22], "gamma_322" = NN.model.refitting.to.final.res$weights[[1]][[1]][,3][23],
                           
                           "beta_4" = NN.model.refitting.to.final.res$weights[[1]][[2]][[5]], 
                           "gamma_40" = NN.model.refitting.to.final.res$weights[[1]][[1]][,4][1], "gamma_41" = NN.model.refitting.to.final.res$weights[[1]][[1]][,4][2], "gamma_42" = NN.model.refitting.to.final.res$weights[[1]][[1]][,4][3], 
                           "gamma_43" = NN.model.refitting.to.final.res$weights[[1]][[1]][,4][4], "gamma_44" = NN.model.refitting.to.final.res$weights[[1]][[1]][,4][5], "gamma_45" = NN.model.refitting.to.final.res$weights[[1]][[1]][,4][6],
                           "gamma_46" = NN.model.refitting.to.final.res$weights[[1]][[1]][,4][7], "gamma_47" = NN.model.refitting.to.final.res$weights[[1]][[1]][,4][8], "gamma_48" = NN.model.refitting.to.final.res$weights[[1]][[1]][,4][9],
                           "gamma_49" = NN.model.refitting.to.final.res$weights[[1]][[1]][,4][10], "gamma_410" = NN.model.refitting.to.final.res$weights[[1]][[1]][,4][11], "gamma_411" = NN.model.refitting.to.final.res$weights[[1]][[1]][,4][12],
                           "gamma_412" = NN.model.refitting.to.final.res$weights[[1]][[1]][,4][13], "gamma_413" = NN.model.refitting.to.final.res$weights[[1]][[1]][,4][14], "gamma_414" = NN.model.refitting.to.final.res$weights[[1]][[1]][,4][15],
                           "gamma_415" = NN.model.refitting.to.final.res$weights[[1]][[1]][,4][16], "gamma_416" = NN.model.refitting.to.final.res$weights[[1]][[1]][,4][17], "gamma_417" = NN.model.refitting.to.final.res$weights[[1]][[1]][,4][18],
                           "gamma_418" = NN.model.refitting.to.final.res$weights[[1]][[1]][,4][19], "gamma_419" = NN.model.refitting.to.final.res$weights[[1]][[1]][,4][20], "gamma_420" = NN.model.refitting.to.final.res$weights[[1]][[1]][,4][21], 
                           "gamma_421" = NN.model.refitting.to.final.res$weights[[1]][[1]][,4][22], "gamma_422" = NN.model.refitting.to.final.res$weights[[1]][[1]][,4][23],
                           
                           "beta_5" = NN.model.refitting.to.final.res$weights[[1]][[2]][[6]], 
                           "gamma_50" = NN.model.refitting.to.final.res$weights[[1]][[1]][,5][1], "gamma_51" = NN.model.refitting.to.final.res$weights[[1]][[1]][,5][2], "gamma_52" = NN.model.refitting.to.final.res$weights[[1]][[1]][,5][3], 
                           "gamma_53" = NN.model.refitting.to.final.res$weights[[1]][[1]][,5][4], "gamma_54" = NN.model.refitting.to.final.res$weights[[1]][[1]][,5][5], "gamma_55" = NN.model.refitting.to.final.res$weights[[1]][[1]][,5][6],
                           "gamma_56" = NN.model.refitting.to.final.res$weights[[1]][[1]][,5][7], "gamma_57" = NN.model.refitting.to.final.res$weights[[1]][[1]][,5][8], "gamma_58" = NN.model.refitting.to.final.res$weights[[1]][[1]][,5][9],
                           "gamma_59" = NN.model.refitting.to.final.res$weights[[1]][[1]][,5][10], "gamma_510" = NN.model.refitting.to.final.res$weights[[1]][[1]][,5][11], "gamma_511" = NN.model.refitting.to.final.res$weights[[1]][[1]][,5][12],
                           "gamma_512" = NN.model.refitting.to.final.res$weights[[1]][[1]][,5][13], "gamma_513" = NN.model.refitting.to.final.res$weights[[1]][[1]][,5][14], "gamma_514" = NN.model.refitting.to.final.res$weights[[1]][[1]][,5][15],
                           "gamma_515" = NN.model.refitting.to.final.res$weights[[1]][[1]][,5][16], "gamma_516" = NN.model.refitting.to.final.res$weights[[1]][[1]][,5][17], "gamma_517" = NN.model.refitting.to.final.res$weights[[1]][[1]][,5][18],
                           "gamma_518" = NN.model.refitting.to.final.res$weights[[1]][[1]][,5][19], "gamma_519" = NN.model.refitting.to.final.res$weights[[1]][[1]][,5][20], "gamma_520" = NN.model.refitting.to.final.res$weights[[1]][[1]][,5][21], 
                           "gamma_521" = NN.model.refitting.to.final.res$weights[[1]][[1]][,5][22], "gamma_522" = NN.model.refitting.to.final.res$weights[[1]][[1]][,5][23],
                           
                           "beta_6" = NN.model.refitting.to.final.res$weights[[1]][[2]][[7]], 
                           "gamma_60" = NN.model.refitting.to.final.res$weights[[1]][[1]][,6][1], "gamma_61" = NN.model.refitting.to.final.res$weights[[1]][[1]][,6][2], "gamma_62" = NN.model.refitting.to.final.res$weights[[1]][[1]][,6][3], 
                           "gamma_63" = NN.model.refitting.to.final.res$weights[[1]][[1]][,6][4], "gamma_64" = NN.model.refitting.to.final.res$weights[[1]][[1]][,6][5], "gamma_65" = NN.model.refitting.to.final.res$weights[[1]][[1]][,6][6],
                           "gamma_66" = NN.model.refitting.to.final.res$weights[[1]][[1]][,6][7], "gamma_67" = NN.model.refitting.to.final.res$weights[[1]][[1]][,6][8], "gamma_68" = NN.model.refitting.to.final.res$weights[[1]][[1]][,6][9],
                           "gamma_69" = NN.model.refitting.to.final.res$weights[[1]][[1]][,6][10], "gamma_610" = NN.model.refitting.to.final.res$weights[[1]][[1]][,6][11], "gamma_611" = NN.model.refitting.to.final.res$weights[[1]][[1]][,6][12],
                           "gamma_612" = NN.model.refitting.to.final.res$weights[[1]][[1]][,6][13], "gamma_613" = NN.model.refitting.to.final.res$weights[[1]][[1]][,6][14], "gamma_614" = NN.model.refitting.to.final.res$weights[[1]][[1]][,6][15],
                           "gamma_615" = NN.model.refitting.to.final.res$weights[[1]][[1]][,6][16], "gamma_616" = NN.model.refitting.to.final.res$weights[[1]][[1]][,6][17], "gamma_617" = NN.model.refitting.to.final.res$weights[[1]][[1]][,6][18],
                           "gamma_618" = NN.model.refitting.to.final.res$weights[[1]][[1]][,6][19], "gamma_619" = NN.model.refitting.to.final.res$weights[[1]][[1]][,6][20], "gamma_620" = NN.model.refitting.to.final.res$weights[[1]][[1]][,6][21], 
                           "gamma_621" = NN.model.refitting.to.final.res$weights[[1]][[1]][,6][22], "gamma_622" = NN.model.refitting.to.final.res$weights[[1]][[1]][,6][23],
                           
                           "beta_7" = NN.model.refitting.to.final.res$weights[[1]][[2]][[8]], 
                           "gamma_70" = NN.model.refitting.to.final.res$weights[[1]][[1]][,7][1], "gamma_71" = NN.model.refitting.to.final.res$weights[[1]][[1]][,7][2], "gamma_72" = NN.model.refitting.to.final.res$weights[[1]][[1]][,7][3], 
                           "gamma_73" = NN.model.refitting.to.final.res$weights[[1]][[1]][,7][4], "gamma_74" = NN.model.refitting.to.final.res$weights[[1]][[1]][,7][5], "gamma_75" = NN.model.refitting.to.final.res$weights[[1]][[1]][,7][6],
                           "gamma_76" = NN.model.refitting.to.final.res$weights[[1]][[1]][,7][7], "gamma_77" = NN.model.refitting.to.final.res$weights[[1]][[1]][,7][8], "gamma_78" = NN.model.refitting.to.final.res$weights[[1]][[1]][,7][9],
                           "gamma_79" = NN.model.refitting.to.final.res$weights[[1]][[1]][,7][10], "gamma_710" = NN.model.refitting.to.final.res$weights[[1]][[1]][,7][11], "gamma_711" = NN.model.refitting.to.final.res$weights[[1]][[1]][,7][12],
                           "gamma_712" = NN.model.refitting.to.final.res$weights[[1]][[1]][,7][13], "gamma_713" = NN.model.refitting.to.final.res$weights[[1]][[1]][,7][14], "gamma_714" = NN.model.refitting.to.final.res$weights[[1]][[1]][,7][15],
                           "gamma_715" = NN.model.refitting.to.final.res$weights[[1]][[1]][,7][16], "gamma_716" = NN.model.refitting.to.final.res$weights[[1]][[1]][,7][17], "gamma_717" = NN.model.refitting.to.final.res$weights[[1]][[1]][,7][18],
                           "gamma_718" = NN.model.refitting.to.final.res$weights[[1]][[1]][,7][19], "gamma_719" = NN.model.refitting.to.final.res$weights[[1]][[1]][,7][20], "gamma_720" = NN.model.refitting.to.final.res$weights[[1]][[1]][,7][21], 
                           "gamma_721" = NN.model.refitting.to.final.res$weights[[1]][[1]][,7][22], "gamma_722" = NN.model.refitting.to.final.res$weights[[1]][[1]][,7][23],
                           
                           "beta_8" = NN.model.refitting.to.final.res$weights[[1]][[2]][[9]], 
                           "gamma_80" = NN.model.refitting.to.final.res$weights[[1]][[1]][,8][1], "gamma_81" = NN.model.refitting.to.final.res$weights[[1]][[1]][,8][2], "gamma_82" = NN.model.refitting.to.final.res$weights[[1]][[1]][,8][3], 
                           "gamma_83" = NN.model.refitting.to.final.res$weights[[1]][[1]][,8][4], "gamma_84" = NN.model.refitting.to.final.res$weights[[1]][[1]][,8][5], "gamma_85" = NN.model.refitting.to.final.res$weights[[1]][[1]][,8][6],
                           "gamma_86" = NN.model.refitting.to.final.res$weights[[1]][[1]][,8][7], "gamma_87" = NN.model.refitting.to.final.res$weights[[1]][[1]][,8][8], "gamma_88" = NN.model.refitting.to.final.res$weights[[1]][[1]][,8][9],
                           "gamma_89" = NN.model.refitting.to.final.res$weights[[1]][[1]][,8][10], "gamma_810" = NN.model.refitting.to.final.res$weights[[1]][[1]][,8][11], "gamma_811" = NN.model.refitting.to.final.res$weights[[1]][[1]][,8][12],
                           "gamma_812" = NN.model.refitting.to.final.res$weights[[1]][[1]][,8][13], "gamma_813" = NN.model.refitting.to.final.res$weights[[1]][[1]][,8][14], "gamma_814" = NN.model.refitting.to.final.res$weights[[1]][[1]][,8][15],
                           "gamma_815" = NN.model.refitting.to.final.res$weights[[1]][[1]][,8][16], "gamma_816" = NN.model.refitting.to.final.res$weights[[1]][[1]][,8][17], "gamma_817" = NN.model.refitting.to.final.res$weights[[1]][[1]][,8][18],
                           "gamma_818" = NN.model.refitting.to.final.res$weights[[1]][[1]][,8][19], "gamma_819" = NN.model.refitting.to.final.res$weights[[1]][[1]][,8][20], "gamma_820" = NN.model.refitting.to.final.res$weights[[1]][[1]][,8][21], 
                           "gamma_821" = NN.model.refitting.to.final.res$weights[[1]][[1]][,8][22], "gamma_822" = NN.model.refitting.to.final.res$weights[[1]][[1]][,8][23],
                           
                           "beta_9" = NN.model.refitting.to.final.res$weights[[1]][[2]][[10]], 
                           "gamma_90" = NN.model.refitting.to.final.res$weights[[1]][[1]][,9][1], "gamma_91" = NN.model.refitting.to.final.res$weights[[1]][[1]][,9][2], "gamma_92" = NN.model.refitting.to.final.res$weights[[1]][[1]][,9][3], 
                           "gamma_93" = NN.model.refitting.to.final.res$weights[[1]][[1]][,9][4], "gamma_94" = NN.model.refitting.to.final.res$weights[[1]][[1]][,9][5], "gamma_95" = NN.model.refitting.to.final.res$weights[[1]][[1]][,9][6],
                           "gamma_96" = NN.model.refitting.to.final.res$weights[[1]][[1]][,9][7], "gamma_97" = NN.model.refitting.to.final.res$weights[[1]][[1]][,9][8], "gamma_98" = NN.model.refitting.to.final.res$weights[[1]][[1]][,9][9],
                           "gamma_99" = NN.model.refitting.to.final.res$weights[[1]][[1]][,9][10], "gamma_910" = NN.model.refitting.to.final.res$weights[[1]][[1]][,9][11], "gamma_911" = NN.model.refitting.to.final.res$weights[[1]][[1]][,9][12],
                           "gamma_912" = NN.model.refitting.to.final.res$weights[[1]][[1]][,9][13], "gamma_913" = NN.model.refitting.to.final.res$weights[[1]][[1]][,9][14], "gamma_914" = NN.model.refitting.to.final.res$weights[[1]][[1]][,9][15],
                           "gamma_915" = NN.model.refitting.to.final.res$weights[[1]][[1]][,9][16], "gamma_916" = NN.model.refitting.to.final.res$weights[[1]][[1]][,9][17], "gamma_917" = NN.model.refitting.to.final.res$weights[[1]][[1]][,9][18],
                           "gamma_918" = NN.model.refitting.to.final.res$weights[[1]][[1]][,9][19], "gamma_919" = NN.model.refitting.to.final.res$weights[[1]][[1]][,9][20], "gamma_920" = NN.model.refitting.to.final.res$weights[[1]][[1]][,9][21], 
                           "gamma_921" = NN.model.refitting.to.final.res$weights[[1]][[1]][,9][22], "gamma_922" = NN.model.refitting.to.final.res$weights[[1]][[1]][,9][23],
                           
                           "beta_10" = NN.model.refitting.to.final.res$weights[[1]][[2]][[11]], 
                           "gamma_100" = NN.model.refitting.to.final.res$weights[[1]][[1]][,10][1], "gamma_101" = NN.model.refitting.to.final.res$weights[[1]][[1]][,10][2], "gamma_102" = NN.model.refitting.to.final.res$weights[[1]][[1]][,10][3], 
                           "gamma_103" = NN.model.refitting.to.final.res$weights[[1]][[1]][,10][4], "gamma_104" = NN.model.refitting.to.final.res$weights[[1]][[1]][,10][5], "gamma_105" = NN.model.refitting.to.final.res$weights[[1]][[1]][,10][6],
                           "gamma_106" = NN.model.refitting.to.final.res$weights[[1]][[1]][,10][7], "gamma_107" = NN.model.refitting.to.final.res$weights[[1]][[1]][,10][8], "gamma_108" = NN.model.refitting.to.final.res$weights[[1]][[1]][,10][9],
                           "gamma_109" = NN.model.refitting.to.final.res$weights[[1]][[1]][,10][10], "gamma_1010" = NN.model.refitting.to.final.res$weights[[1]][[1]][,10][11], "gamma_1011" = NN.model.refitting.to.final.res$weights[[1]][[1]][,10][12],
                           "gamma_1012" = NN.model.refitting.to.final.res$weights[[1]][[1]][,10][13], "gamma_1013" = NN.model.refitting.to.final.res$weights[[1]][[1]][,10][14], "gamma_1014" = NN.model.refitting.to.final.res$weights[[1]][[1]][,10][15],
                           "gamma_1015" = NN.model.refitting.to.final.res$weights[[1]][[1]][,10][16], "gamma_1016" = NN.model.refitting.to.final.res$weights[[1]][[1]][,10][17], "gamma_1017" = NN.model.refitting.to.final.res$weights[[1]][[1]][,10][18],
                           "gamma_1018" = NN.model.refitting.to.final.res$weights[[1]][[1]][,10][19], "gamma_1019" = NN.model.refitting.to.final.res$weights[[1]][[1]][,10][20], "gamma_1020" = NN.model.refitting.to.final.res$weights[[1]][[1]][,10][21], 
                           "gamma_1021" = NN.model.refitting.to.final.res$weights[[1]][[1]][,10][22], "gamma_1022" = NN.model.refitting.to.final.res$weights[[1]][[1]][,10][23] )
                           
                           
                           

#### Predicted value for RV_[T+1d]^(d), where T = nrow(RV.data)-i.
  predicted.values.for.RV.Tplus1d.d  <- AR22.model.t.coeffi*Rv.d.x1[nrow(RV.data)-i] + AR22.model.t.1.coeffi*Rv.d.x1[nrow(RV.data)-(i+1)]+
    AR22.model.t.2.coeffi*Rv.d.x1[nrow(RV.data)-(i+2)]+AR22.model.t.3.coeffi*Rv.d.x1[nrow(RV.data)-(i+3)]+AR22.model.t.4.coeffi*Rv.d.x1[nrow(RV.data)-(i+4)]+
    AR22.model.t.5.coeffi*Rv.d.x1[nrow(RV.data)-(i+5)]+AR22.model.t.6.coeffi*Rv.d.x1[nrow(RV.data)-(i+6)]+
    AR22.model.t.7.coeffi*Rv.d.x1[nrow(RV.data)-(i+7)]+AR22.model.t.8.coeffi*Rv.d.x1[nrow(RV.data)-(i+8)]+AR22.model.t.9.coeffi*Rv.d.x1[nrow(RV.data)-(i+9)]
  +AR22.model.t.10.coeffi*Rv.d.x1[nrow(RV.data)-(i+10)]+AR22.model.t.11.coeffi*Rv.d.x1[nrow(RV.data)-(i+11)]+AR22.model.t.12.coeffi*Rv.d.x1[nrow(RV.data)-(i+12)]
  +AR22.model.t.13.coeffi*Rv.d.x1[nrow(RV.data)-(i+13)]+AR22.model.t.14.coeffi*Rv.d.x1[nrow(RV.data)-(i+14)]+AR22.model.t.15.coeffi*Rv.d.x1[nrow(RV.data)-(i+15)]
  +AR22.model.t.16.coeffi*Rv.d.x1[nrow(RV.data)-(i+16)]+AR22.model.t.17.coeffi*Rv.d.x1[nrow(RV.data)-(i+17)]+AR22.model.t.18.coeffi*Rv.d.x1[nrow(RV.data)-(i+18)]
  +AR22.model.t.19.coeffi*Rv.d.x1[nrow(RV.data)-(i+19)]+AR22.model.t.20.coeffi*Rv.d.x1[nrow(RV.data)-(i+20)]
  +AR22.model.t.21.coeffi*Rv.d.x1[nrow(RV.data)-(i+21)]+NN.model.refitting.to.final.res$weights[[1]][[2]][[1]]+NN.model.refitting.to.final.res$weights[[1]][[2]][[2]]*a1+NN.model.refitting.to.final.res$weights[[1]][[2]][[3]]*a2+
    NN.model.refitting.to.final.res$weights[[1]][[2]][[4]]*a3+NN.model.refitting.to.final.res$weights[[1]][[2]][[5]]*a4+NN.model.refitting.to.final.res$weights[[1]][[2]][[6]]*a5
  +NN.model.refitting.to.final.res$weights[[1]][[2]][[7]]*a6+NN.model.refitting.to.final.res$weights[[1]][[2]][[8]]*a7+NN.model.refitting.to.final.res$weights[[1]][[2]][[9]]*a8
  +NN.model.refitting.to.final.res$weights[[1]][[2]][[10]]*a9+NN.model.refitting.to.final.res$weights[[1]][[2]][[11]]*a10
  
  
  if(printcoeffi=="YES"){
    print(coefficient.list)
  }
  
  else if(printcoeffi=="NO"){
    print(predicted.values.for.RV.Tplus1d.d)
  }
  
  else if(printcoeffi=="BOTH"){
    print(coefficient.list)
    print(predicted.values.for.RV.Tplus1d.d)
  }
  
  
  else
    print("You might have a typo error in 'printcoeffi' argument")
  
}


# HAR_infty_NN.tanh(KOSPI.0615.data.RV,100,200,"YES")




##### HAR-infty-NN forecasting function.

HAR_infty_NN_forecast <- function(RV.data,i,r,acti.fun){
  
  if(acti.fun=="logistic"){
    
    ########### Predicted value for RV_[T+1d]^(d), where T = nrow(RV.data)-i
    
    HAR_infty_NN.sig(KOSPI.0615.data.RV,i,r,"NO")
    
  }
  else if(acti.fun=="tanh"){
    
    ########### Predicted value for RV_[T+1d]^(d), where T = nrow(RV.data)-i 
    
    HAR_infty_NN.tanh(KOSPI.0615.data.RV,i,r,"NO")
  }
  
  else
    print("You might have a typo error in 'acti.fun' argument")
}


#HAR_infty_NN_forecast(KOSPI.0615.data.RV,100,200,"logistic")

#HAR_infty_NN_forecast(KOSPI.0615.data.RV,100,200,"tanh")

#HAR_infty_NN_forecast(KOSPI.0615.data.RV,100,200,"asdf")







########### HAR-AR(22)-NN model : the third NN-based HAR model. 

########### The optimal q is the same as the above case. (sigmoid=logistic : q=5) & (tanh : q=10) for KOSPI RV series.


HAR_AR22_NN.sig <- function(RV.data,i,r,printcoeffi){
  
  Rv.d.y <- as.numeric(as.vector(RV.data[c(2:((nrow(RV.data)-i)+1)),2])) ## from 2 (daily time point) ~ (nrow(RV.data)-100)+1 (daily daily time point). 
  
  Rv.d.x1 <- as.numeric(as.vector(RV.data[c(1:(nrow(RV.data)-i)),2])) ## from 1 (daily time point) ~ (nrow(RV.data)-100) (daily daily time point). 
  
  ## Data handling for AR(22) model fitting.
  
  Rv.d.y2 <- Rv.d.y[23:(nrow(RV.data)-i)]  # (t+1d) daily time point RV series data.
  
  Rv.d.x1.2 <- Rv.d.x1[23:(nrow(RV.data)-i)] # (t) daily time point RV series data.
  
  ## Weekly, monthly RV.
  
  Rv.w <- numeric((nrow(RV.data)-i))
  for(j in 23:(nrow(RV.data)-i)){
    Rv.w[j] <- (Rv.d.x1[j]+Rv.d.x1[j-1]+Rv.d.x1[j-2]+Rv.d.x1[j-3]+Rv.d.x1[j-4])/5
  }
  
  
  Rv.m <- numeric((nrow(RV.data)-i)) 
  for(q in 23:(nrow(RV.data)-i)){
    Rv.m[q] <- (Rv.d.x1[q]+Rv.d.x1[q-1]+Rv.d.x1[q-2]+Rv.d.x1[q-3]
                         +Rv.d.x1[q-4]+Rv.d.x1[q-5]+Rv.d.x1[q-6]+Rv.d.x1[q-7]
                         +Rv.d.x1[q-8]+Rv.d.x1[q-9]+Rv.d.x1[q-10]+Rv.d.x1[q-11]
                         +Rv.d.x1[q-12]+Rv.d.x1[q-13]+Rv.d.x1[q-14]+Rv.d.x1[q-15]
                         +Rv.d.x1[q-16]+Rv.d.x1[q-17]+Rv.d.x1[q-18]+Rv.d.x1[q-19]
                         +Rv.d.x1[q-20]+Rv.d.x1[q-21])/22
  }
  
  Rv.w.2 <- Rv.w[23:i]
  Rv.m.2 <- Rv.m[23:i]
  
  
########### STEP 1 : Initial AR(22) model fitting to data set. 
  
  initial.AR22.model.fit <- arima(Rv.d.y2,order=c(22,0,0),method="CSS") 
  
  initial.AR22.model.coeffi <- initial.AR22.model.fit$coef
  
  initial.AR22.model.coeffi <- as.vector(initial.AR22.model.coeffi)
  
  # (t-1d) daily time point series data
  Rv.d.x1.2.1 <- c(Rv.d.x1[22],Rv.d.x1.2[-length(Rv.d.x1.2)])
  
  # (t-2d) daily time point series data
  Rv.d.x1.2.2 <- c(Rv.d.x1[21],Rv.d.x1[22],Rv.d.x1.2[-(length(Rv.d.x1.2)-1):-length(Rv.d.x1.2)])
  
  # (t-3d) daily time point series data
  Rv.d.x1.2.3 <- c(Rv.d.x1[20],Rv.d.x1[21],Rv.d.x1[22],Rv.d.x1.2[-(length(Rv.d.x1.2)-2):-length(Rv.d.x1.2)])
  
  # (t-4d) daily time point series data
  Rv.d.x1.2.4 <- c(Rv.d.x1[19],Rv.d.x1[20],Rv.d.x1[21],Rv.d.x1[22],Rv.d.x1.2[-(length(Rv.d.x1.2)-3):-length(Rv.d.x1.2)])
  
  # (t-5d) daily time point series data
  Rv.d.x1.2.5 <- c(Rv.d.x1[18],Rv.d.x1[19],Rv.d.x1[20],Rv.d.x1[21],Rv.d.x1[22],Rv.d.x1.2[-(length(Rv.d.x1.2)-4):-length(Rv.d.x1.2)])
  
  # (t-6d) daily time point series data
  Rv.d.x1.2.6 <- c(Rv.d.x1[17],Rv.d.x1[18],Rv.d.x1[19],Rv.d.x1[20],Rv.d.x1[21],Rv.d.x1[22],Rv.d.x1.2[-(length(Rv.d.x1.2)-5):-length(Rv.d.x1.2)])
  
  # (t-7d) daily time point series data
  Rv.d.x1.2.7 <- c(Rv.d.x1[16],Rv.d.x1[17],Rv.d.x1[18],Rv.d.x1[19],Rv.d.x1[20],Rv.d.x1[21],Rv.d.x1[22],Rv.d.x1.2[-(length(Rv.d.x1.2)-6):-length(Rv.d.x1.2)])
  
  # (t-8d) daily time point series data
  Rv.d.x1.2.8 <- c(Rv.d.x1[15],Rv.d.x1[16],Rv.d.x1[17],Rv.d.x1[18],Rv.d.x1[19],Rv.d.x1[20],Rv.d.x1[21],Rv.d.x1[22],Rv.d.x1.2[-(length(Rv.d.x1.2)-7):-length(Rv.d.x1.2)])
  
  # (t-9d) daily time point series data
  Rv.d.x1.2.9 <- c(Rv.d.x1[14],Rv.d.x1[15],Rv.d.x1[16],Rv.d.x1[17],Rv.d.x1[18],Rv.d.x1[19],Rv.d.x1[20],Rv.d.x1[21],Rv.d.x1[22],Rv.d.x1.2[-(length(Rv.d.x1.2)-8):-length(Rv.d.x1.2)])
  
  # (t-10d) daily time point series data
  Rv.d.x1.2.10 <- c(Rv.d.x1[13],Rv.d.x1[14],Rv.d.x1[15],Rv.d.x1[16],Rv.d.x1[17],Rv.d.x1[18],Rv.d.x1[19],Rv.d.x1[20],Rv.d.x1[21],Rv.d.x1[22],Rv.d.x1.2[-(length(Rv.d.x1.2)-9):-length(Rv.d.x1.2)])
  
  # (t-11d) daily time point series data
  Rv.d.x1.2.11 <- c(Rv.d.x1[12],Rv.d.x1[13],Rv.d.x1[14],Rv.d.x1[15],Rv.d.x1[16],Rv.d.x1[17],Rv.d.x1[18],Rv.d.x1[19],Rv.d.x1[20],Rv.d.x1[21],Rv.d.x1[22],Rv.d.x1.2[-(length(Rv.d.x1.2)-10):-length(Rv.d.x1.2)])
  
  # (t-12d) daily time point series data
  Rv.d.x1.2.12 <- c(Rv.d.x1[11],Rv.d.x1[12],Rv.d.x1[13],Rv.d.x1[14],Rv.d.x1[15],Rv.d.x1[16],Rv.d.x1[17],Rv.d.x1[18],Rv.d.x1[19],Rv.d.x1[20],Rv.d.x1[21],Rv.d.x1[22],Rv.d.x1.2[-(length(Rv.d.x1.2)-11):-length(Rv.d.x1.2)])
  
  # (t-13d) daily time point series data
  Rv.d.x1.2.13 <- c(Rv.d.x1[10],Rv.d.x1[11],Rv.d.x1[12],Rv.d.x1[13],Rv.d.x1[14],Rv.d.x1[15],Rv.d.x1[16],Rv.d.x1[17],Rv.d.x1[18],Rv.d.x1[19],Rv.d.x1[20],Rv.d.x1[21],Rv.d.x1[22],Rv.d.x1.2[-(length(Rv.d.x1.2)-12):-length(Rv.d.x1.2)])
  
  # (t-14d) daily time point series data
  Rv.d.x1.2.14 <- c(Rv.d.x1[9],Rv.d.x1[10],Rv.d.x1[11],Rv.d.x1[12],Rv.d.x1[13],Rv.d.x1[14],Rv.d.x1[15],Rv.d.x1[16],Rv.d.x1[17],Rv.d.x1[18],Rv.d.x1[19],Rv.d.x1[20],Rv.d.x1[21],Rv.d.x1[22],Rv.d.x1.2[-(length(Rv.d.x1.2)-13):-length(Rv.d.x1.2)])
  
  # (t-15d) daily time point series data
  Rv.d.x1.2.15 <- c(Rv.d.x1[8],Rv.d.x1[9],Rv.d.x1[10],Rv.d.x1[11],Rv.d.x1[12],Rv.d.x1[13],Rv.d.x1[14],Rv.d.x1[15],Rv.d.x1[16],Rv.d.x1[17],Rv.d.x1[18],Rv.d.x1[19],Rv.d.x1[20],Rv.d.x1[21],Rv.d.x1[22],Rv.d.x1.2[-(length(Rv.d.x1.2)-14):-length(Rv.d.x1.2)])
  
  # (t-16d) daily time point series data
  Rv.d.x1.2.16 <- c(Rv.d.x1[7],Rv.d.x1[8],Rv.d.x1[9],Rv.d.x1[10],Rv.d.x1[11],Rv.d.x1[12],Rv.d.x1[13],Rv.d.x1[14],Rv.d.x1[15],Rv.d.x1[16],Rv.d.x1[17],Rv.d.x1[18],Rv.d.x1[19],Rv.d.x1[20],Rv.d.x1[21],Rv.d.x1[22],Rv.d.x1.2[-(length(Rv.d.x1.2)-15):-length(Rv.d.x1.2)])
  
  # (t-17d) daily time point series data
  Rv.d.x1.2.17 <- c(Rv.d.x1[6],Rv.d.x1[7],Rv.d.x1[8],Rv.d.x1[9],Rv.d.x1[10],Rv.d.x1[11],Rv.d.x1[12],Rv.d.x1[13],Rv.d.x1[14],Rv.d.x1[15],Rv.d.x1[16],Rv.d.x1[17],Rv.d.x1[18],Rv.d.x1[19],Rv.d.x1[20],Rv.d.x1[21],Rv.d.x1[22],Rv.d.x1.2[-(length(Rv.d.x1.2)-16):-length(Rv.d.x1.2)])
  
  # (t-18d) daily time point series data
  Rv.d.x1.2.18 <- c(Rv.d.x1[5],Rv.d.x1[6],Rv.d.x1[7],Rv.d.x1[8],Rv.d.x1[9],Rv.d.x1[10],Rv.d.x1[11],Rv.d.x1[12],Rv.d.x1[13],Rv.d.x1[14],Rv.d.x1[15],Rv.d.x1[16],Rv.d.x1[17],Rv.d.x1[18],Rv.d.x1[19],Rv.d.x1[20],Rv.d.x1[21],Rv.d.x1[22],Rv.d.x1.2[-(length(Rv.d.x1.2)-17):-length(Rv.d.x1.2)])
  
  # (t-19d) daily time point series data
  Rv.d.x1.2.19 <- c(Rv.d.x1[4],Rv.d.x1[5],Rv.d.x1[6],Rv.d.x1[7],Rv.d.x1[8],Rv.d.x1[9],Rv.d.x1[10],Rv.d.x1[11],Rv.d.x1[12],Rv.d.x1[13],Rv.d.x1[14],Rv.d.x1[15],Rv.d.x1[16],Rv.d.x1[17],Rv.d.x1[18],Rv.d.x1[19],Rv.d.x1[20],Rv.d.x1[21],Rv.d.x1[22],Rv.d.x1.2[-(length(Rv.d.x1.2)-18):-length(Rv.d.x1.2)])
  
  # (t-20d) daily time point series data
  Rv.d.x1.2.20 <- c(Rv.d.x1[3],Rv.d.x1[4],Rv.d.x1[5],Rv.d.x1[6],Rv.d.x1[7],Rv.d.x1[8],Rv.d.x1[9],Rv.d.x1[10],Rv.d.x1[11],Rv.d.x1[12],Rv.d.x1[13],Rv.d.x1[14],Rv.d.x1[15],Rv.d.x1[16],Rv.d.x1[17],Rv.d.x1[18],Rv.d.x1[19],Rv.d.x1[20],Rv.d.x1[21],Rv.d.x1[22],Rv.d.x1.2[-(length(Rv.d.x1.2)-19):-length(Rv.d.x1.2)])
  
  # (t-21d) daily time point series data
  Rv.d.x1.2.21 <- c(Rv.d.x1[2],Rv.d.x1[3],Rv.d.x1[4],Rv.d.x1[5],Rv.d.x1[6],Rv.d.x1[7],Rv.d.x1[8],Rv.d.x1[9],Rv.d.x1[10],Rv.d.x1[11],Rv.d.x1[12],Rv.d.x1[13],Rv.d.x1[14],Rv.d.x1[15],Rv.d.x1[16],Rv.d.x1[17],Rv.d.x1[18],Rv.d.x1[19],Rv.d.x1[20],Rv.d.x1[21],Rv.d.x1[22],Rv.d.x1.2[-(length(Rv.d.x1.2)-20):-length(Rv.d.x1.2)])
  
  
  
  ## initial residual
  
  initial.res <-(Rv.d.y2-initial.AR22.model.coeffi[1]*Rv.d.x1.2-initial.AR22.model.coeffi[2]*Rv.d.x1.2.1-initial.AR22.model.coeffi[3]*Rv.d.x1.2.2
                 -initial.AR22.model.coeffi[4]*Rv.d.x1.2.3-initial.AR22.model.coeffi[5]*Rv.d.x1.2.4-initial.AR22.model.coeffi[6]*Rv.d.x1.2.5-initial.AR22.model.coeffi[7]*Rv.d.x1.2.6
                 -initial.AR22.model.coeffi[8]*Rv.d.x1.2.7-initial.AR22.model.coeffi[9]*Rv.d.x1.2.8-initial.AR22.model.coeffi[10]*Rv.d.x1.2.9-initial.AR22.model.coeffi[11]*Rv.d.x1.2.10-initial.AR22.model.coeffi[12]*Rv.d.x1.2.11
                 -initial.AR22.model.coeffi[13]*Rv.d.x1.2.12-initial.AR22.model.coeffi[14]*Rv.d.x1.2.13-initial.AR22.model.coeffi[15]*Rv.d.x1.2.14-initial.AR22.model.coeffi[16]*Rv.d.x1.2.15
                 -initial.AR22.model.coeffi[17]*Rv.d.x1.2.16-initial.AR22.model.coeffi[18]*Rv.d.x1.2.17-initial.AR22.model.coeffi[19]*Rv.d.x1.2.18-initial.AR22.model.coeffi[20]*Rv.d.x1.2.19-initial.AR22.model.coeffi[21]*Rv.d.x1.2.20
                 -initial.AR22.model.coeffi[22]*Rv.d.x1.2.21)
  
  
  RVcombined.data <- cbind(initial.res,Rv.d.x1.2,Rv.w.2,Rv.m.2)
  
########### STEP 2 : Initial NN fitting to initial residual set. 
  
  ## NN model fitting to initial res data.
  
  NN.model.fitting.to.initial.res <- neuralnet(initial.res~Rv.d.x1.2+Rv.w.2+Rv.m.2,data=RVcombined.data,
                                               hidden=5,linear.output=T)
  
  
  ## The AR(22) model coefficient value corresponding to RV_[t] (numeric vector to store the updated value)
  AR22.model.coeffi.beta01.vector <- numeric(r)
  u.AR22.model.coeffi.beta01.vector <- numeric(r)
  
  ## The AR(22) model coefficient value corresponding to RV_[t-1d]
  AR22.model.coeffi.beta02.vector <- numeric(r)
  u.AR22.model.coeffi.beta02.vector <- numeric(r)
  
  ## The same to RV_[t-2d]
  AR22.model.coeffi.beta03.vector <- numeric(r)
  u.AR22.model.coeffi.beta03.vector <- numeric(r)
  
  ## The same to RV_[t-3d]
  AR22.model.coeffi.beta04.vector <- numeric(r)
  u.AR22.model.coeffi.beta04.vector <- numeric(r)
  
  ## The same to RV_[t-4d]
  AR22.model.coeffi.beta05.vector <- numeric(r)
  u.AR22.model.coeffi.beta05.vector <- numeric(r)
  
  ## The same to RV_[t-5d]
  AR22.model.coeffi.beta06.vector <- numeric(r)
  u.AR22.model.coeffi.beta06.vector <- numeric(r)
  
  ## The same to RV_[t-6d]
  AR22.model.coeffi.beta07.vector <- numeric(r)
  u.AR22.model.coeffi.beta07.vector <- numeric(r)
  
  ## The same to RV_[t-7d]
  AR22.model.coeffi.beta08.vector <- numeric(r)
  u.AR22.model.coeffi.beta08.vector <- numeric(r)
  
  ## The same to RV_[t-8d]
  AR22.model.coeffi.beta09.vector <- numeric(r)
  u.AR22.model.coeffi.beta09.vector <- numeric(r)
  
  ## The same to RV_[t-9d]
  AR22.model.coeffi.beta010.vector <- numeric(r)
  u.AR22.model.coeffi.beta010.vector <- numeric(r)
  
  ## The same to RV_[t-10d]
  AR22.model.coeffi.beta011.vector <- numeric(r)
  u.AR22.model.coeffi.beta011.vector <- numeric(r)
  
  ## The same to RV_[t-11d]
  AR22.model.coeffi.beta012.vector <- numeric(r)
  u.AR22.model.coeffi.beta012.vector <- numeric(r)
  
  ## The same to RV_[t-12d]
  AR22.model.coeffi.beta013.vector <- numeric(r)
  u.AR22.model.coeffi.beta013.vector <- numeric(r)
  
  ## The same to RV_[t-13d]
  AR22.model.coeffi.beta014.vector <- numeric(r)
  u.AR22.model.coeffi.beta014.vector <- numeric(r)
  
  ## The same to RV_[t-14d]
  AR22.model.coeffi.beta015.vector <- numeric(r)
  u.AR22.model.coeffi.beta015.vector <- numeric(r)
  
  ## The same to RV_[t-15d]
  AR22.model.coeffi.beta016.vector <- numeric(r)
  u.AR22.model.coeffi.beta016.vector <- numeric(r)
  
  ## The same to RV_[t-16d]
  AR22.model.coeffi.beta017.vector <- numeric(r)
  u.AR22.model.coeffi.beta017.vector <- numeric(r)
  
  ## The same to RV_[t-17d]
  AR22.model.coeffi.beta018.vector <- numeric(r)
  u.AR22.model.coeffi.beta018.vector <- numeric(r)
  
  ## The same to RV_[t-18d]
  AR22.model.coeffi.beta019.vector <- numeric(r)
  u.AR22.model.coeffi.beta019.vector <- numeric(r)
  
  ## The same to RV_[t-19d]
  AR22.model.coeffi.beta020.vector <- numeric(r)
  u.AR22.model.coeffi.beta020.vector <- numeric(r)
  
  ## The same to RV_[t-20d]
  AR22.model.coeffi.beta021.vector <- numeric(r)
  u.AR22.model.coeffi.beta021.vector <- numeric(r)
  
  ## The same to RV_[t-21d]
  AR22.model.coeffi.beta022.vector <- numeric(r)
  u.AR22.model.coeffi.beta022.vector <- numeric(r)
  
  
  
  
  for(p in 1:r){
    
    
########### STEP 3 : refit.AR22.model to ( Rv_[t+1d](d) - fitted.value of 'NN.model.fitting.to.initial.res' )       
    
    refit.AR22.model <- arima((Rv.d.y2 - NN.model.fitting.to.initial.res$net.result[[1]]),order=c(22,0,0),method="CSS")
    
    ### coefficient
    
    refit.AR22.model.coeffi <-(refit.AR22.model$coef)
    refit.AR22.model.coeffi <- as.vector(refit.AR22.model.coeffi)
    refit.AR22.model.coeffi ## 
    
    
    ## store AR(22) Coefficients obtained from STEP 3
    
    AR22.model.coeffi.beta01.vector[p] <- refit.AR22.model.coeffi[1]
    
    AR22.model.coeffi.beta02.vector[p] <- refit.AR22.model.coeffi[2]
    
    AR22.model.coeffi.beta03.vector[p] <- refit.AR22.model.coeffi[3]
    
    AR22.model.coeffi.beta04.vector[p] <- refit.AR22.model.coeffi[4]
    
    AR22.model.coeffi.beta05.vector[p] <- refit.AR22.model.coeffi[5]
    
    AR22.model.coeffi.beta06.vector[p] <- refit.AR22.model.coeffi[6]
    
    AR22.model.coeffi.beta07.vector[p] <- refit.AR22.model.coeffi[7]
    
    AR22.model.coeffi.beta08.vector[p] <- refit.AR22.model.coeffi[8]
    
    AR22.model.coeffi.beta09.vector[p] <- refit.AR22.model.coeffi[9]
    
    AR22.model.coeffi.beta010.vector[p] <- refit.AR22.model.coeffi[10]
    
    AR22.model.coeffi.beta011.vector[p] <- refit.AR22.model.coeffi[11]
    
    AR22.model.coeffi.beta012.vector[p] <- refit.AR22.model.coeffi[12]
    
    AR22.model.coeffi.beta013.vector[p] <- refit.AR22.model.coeffi[13]
    
    AR22.model.coeffi.beta014.vector[p] <- refit.AR22.model.coeffi[14]
    
    AR22.model.coeffi.beta015.vector[p] <- refit.AR22.model.coeffi[15]
    
    AR22.model.coeffi.beta016.vector[p] <- refit.AR22.model.coeffi[16]
    
    AR22.model.coeffi.beta017.vector[p] <- refit.AR22.model.coeffi[17]
    
    AR22.model.coeffi.beta018.vector[p] <- refit.AR22.model.coeffi[18]
    
    AR22.model.coeffi.beta019.vector[p] <- refit.AR22.model.coeffi[19]
    
    AR22.model.coeffi.beta020.vector[p] <- refit.AR22.model.coeffi[20]
    
    AR22.model.coeffi.beta021.vector[p] <- refit.AR22.model.coeffi[21]
    
    AR22.model.coeffi.beta022.vector[p] <- refit.AR22.model.coeffi[22]
    
    
########### STEP 4 : Fit NN again to second.res data.
    
    
    # second.res
    second.res <-(Rv.d.y2-AR22.model.coeffi.beta01.vector[p]*Rv.d.x1.2-AR22.model.coeffi.beta02.vector[p]*Rv.d.x1.2.1-AR22.model.coeffi.beta03.vector[p]*Rv.d.x1.2.2-AR22.model.coeffi.beta04.vector[p]*Rv.d.x1.2.3-AR22.model.coeffi.beta05.vector[p]*Rv.d.x1.2.4-AR22.model.coeffi.beta06.vector[p]*Rv.d.x1.2.5-AR22.model.coeffi.beta07.vector[p]*Rv.d.x1.2.6-AR22.model.coeffi.beta08.vector[p]*Rv.d.x1.2.7
                  -AR22.model.coeffi.beta09.vector[p]*Rv.d.x1.2.8-AR22.model.coeffi.beta010.vector[p]*Rv.d.x1.2.9-AR22.model.coeffi.beta011.vector[p]*Rv.d.x1.2.10-AR22.model.coeffi.beta012.vector[p]*Rv.d.x1.2.11
                  -AR22.model.coeffi.beta013.vector[p]*Rv.d.x1.2.12-AR22.model.coeffi.beta014.vector[p]*Rv.d.x1.2.13-AR22.model.coeffi.beta015.vector[p]*Rv.d.x1.2.14-AR22.model.coeffi.beta016.vector[p]*Rv.d.x1.2.15
                  -AR22.model.coeffi.beta017.vector[p]*Rv.d.x1.2.16-AR22.model.coeffi.beta018.vector[p]*Rv.d.x1.2.17-AR22.model.coeffi.beta019.vector[p]*Rv.d.x1.2.18-AR22.model.coeffi.beta020.vector[p]*Rv.d.x1.2.19-AR22.model.coeffi.beta021.vector[p]*Rv.d.x1.2.20
                  -AR22.model.coeffi.beta022.vector[p]*Rv.d.x1.2.21)
    
    
    RVcombined.data2 <- cbind(second.res,Rv.d.x1.2,Rv.w.2,Rv.m.2)
    
    
    ## re-fit NN model to second.res data.
    refit.NN.model.to.second.res <- neuralnet(second.res~Rv.d.x1.2+Rv.w.2+Rv.m.2,data=RVcombined.data2,
                                              hidden=5,linear.output=T)
    
    
    
############ return to the STEP 3 : rerefit.AR22.model to ( Rv_[t+1d](d) - fitted.value of 'refit.NN.model.to.second.res' ) 
    
    rere.AR22.model.fit <- arima((Rv.d.y2 - refit.NN.model.to.second.res$net.result[[1]]),order=c(22,0,0),method="CSS")
    
    ### coefficient
    
    rere.AR22.model.coeffi <-(rere.AR22.model.fit$coef)
    rere.AR22.model.coeffi <- as.vector(rere.AR22.model.coeffi)
    
    ## store updated AR(22) Coefficients obtained from 'return to the STEP 3'.
    
    u.AR22.model.coeffi.beta01.vector[p] <- rere.AR22.model.coeffi[1] 
    
    u.AR22.model.coeffi.beta02.vector[p] <- rere.AR22.model.coeffi[2] 
    
    u.AR22.model.coeffi.beta03.vector[p] <- rere.AR22.model.coeffi[3] 
    
    u.AR22.model.coeffi.beta04.vector[p] <- rere.AR22.model.coeffi[4] 
    
    u.AR22.model.coeffi.beta05.vector[p] <- rere.AR22.model.coeffi[5] 
    
    u.AR22.model.coeffi.beta06.vector[p] <- rere.AR22.model.coeffi[6] 
    
    u.AR22.model.coeffi.beta07.vector[p] <- rere.AR22.model.coeffi[7] 
    
    u.AR22.model.coeffi.beta08.vector[p] <- rere.AR22.model.coeffi[8] 
    
    u.AR22.model.coeffi.beta09.vector[p] <- rere.AR22.model.coeffi[9] 
    
    u.AR22.model.coeffi.beta010.vector[p] <- rere.AR22.model.coeffi[10] 
    
    u.AR22.model.coeffi.beta011.vector[p] <- rere.AR22.model.coeffi[11] 
    
    u.AR22.model.coeffi.beta012.vector[p] <- rere.AR22.model.coeffi[12] 
    
    u.AR22.model.coeffi.beta013.vector[p] <- rere.AR22.model.coeffi[13] 
    
    u.AR22.model.coeffi.beta014.vector[p] <- rere.AR22.model.coeffi[14] 
    
    u.AR22.model.coeffi.beta015.vector[p] <- rere.AR22.model.coeffi[15] 
    
    u.AR22.model.coeffi.beta016.vector[p] <- rere.AR22.model.coeffi[16] 
    
    u.AR22.model.coeffi.beta017.vector[p] <- rere.AR22.model.coeffi[17] 
    
    u.AR22.model.coeffi.beta018.vector[p] <- rere.AR22.model.coeffi[18] 
    
    u.AR22.model.coeffi.beta019.vector[p] <- rere.AR22.model.coeffi[19] 
    
    u.AR22.model.coeffi.beta020.vector[p] <- rere.AR22.model.coeffi[20] 
    
    u.AR22.model.coeffi.beta021.vector[p] <- rere.AR22.model.coeffi[21] 
    
    u.AR22.model.coeffi.beta022.vector[p] <- rere.AR22.model.coeffi[22] 
    
    
    ## 
    
    diff1=abs(AR22.model.coeffi.beta01.vector[p]-u.AR22.model.coeffi.beta01.vector[p])
    diff2=abs(AR22.model.coeffi.beta02.vector[p]-u.AR22.model.coeffi.beta02.vector[p])
    diff3=abs(AR22.model.coeffi.beta03.vector[p]-u.AR22.model.coeffi.beta03.vector[p])
    diff4=abs(AR22.model.coeffi.beta04.vector[p]-u.AR22.model.coeffi.beta04.vector[p])
    diff5=abs(AR22.model.coeffi.beta05.vector[p]-u.AR22.model.coeffi.beta05.vector[p])
    diff6=abs(AR22.model.coeffi.beta06.vector[p]-u.AR22.model.coeffi.beta06.vector[p])
    diff7=abs(AR22.model.coeffi.beta07.vector[p]-u.AR22.model.coeffi.beta07.vector[p])
    diff8=abs(AR22.model.coeffi.beta08.vector[p]-u.AR22.model.coeffi.beta08.vector[p])
    diff9=abs(AR22.model.coeffi.beta09.vector[p]-u.AR22.model.coeffi.beta09.vector[p])
    diff10=abs(AR22.model.coeffi.beta010.vector[p]-u.AR22.model.coeffi.beta010.vector[p])
    diff11=abs(AR22.model.coeffi.beta011.vector[p]-u.AR22.model.coeffi.beta011.vector[p])
    diff12=abs(AR22.model.coeffi.beta012.vector[p]-u.AR22.model.coeffi.beta012.vector[p])
    diff13=abs(AR22.model.coeffi.beta013.vector[p]-u.AR22.model.coeffi.beta013.vector[p])
    diff14=abs(AR22.model.coeffi.beta014.vector[p]-u.AR22.model.coeffi.beta014.vector[p])
    diff15=abs(AR22.model.coeffi.beta015.vector[p]-u.AR22.model.coeffi.beta015.vector[p])
    diff16=abs(AR22.model.coeffi.beta016.vector[p]-u.AR22.model.coeffi.beta016.vector[p])
    diff17=abs(AR22.model.coeffi.beta017.vector[p]-u.AR22.model.coeffi.beta017.vector[p])
    diff18=abs(AR22.model.coeffi.beta018.vector[p]-u.AR22.model.coeffi.beta018.vector[p])
    diff19=abs(AR22.model.coeffi.beta019.vector[p]-u.AR22.model.coeffi.beta019.vector[p])
    diff20=abs(AR22.model.coeffi.beta020.vector[p]-u.AR22.model.coeffi.beta020.vector[p])
    diff21=abs(AR22.model.coeffi.beta021.vector[p]-u.AR22.model.coeffi.beta021.vector[p])
    diff22=abs(AR22.model.coeffi.beta022.vector[p]-u.AR22.model.coeffi.beta022.vector[p])
    
    
    
    
    if( sum(c(diff1,diff2,diff3,diff4,diff5,diff6,diff7,diff8,diff9,diff10,diff11,diff12,diff13,diff14,diff15,diff16,diff17,diff18,diff19,diff20,diff21,diff22)<0.05)==22 )
    {
      
      # convergent coefficient corresponding to RV_[t] in AR(22) model part of HAR-infty-NN model. 
      AR22.model.t.coeffi <- as.numeric(AR22.model.coeffi.beta01.vector[p])
      
      # The same to RV_[t-1d]
      AR22.model.t.1.coeffi <- as.numeric(AR22.model.coeffi.beta02.vector[p])
      
      # The same to RV_[t-2d]
      AR22.model.t.2.coeffi <- as.numeric(AR22.model.coeffi.beta03.vector[p])
      
      # The same to RV_[t-3d]
      AR22.model.t.3.coeffi <- as.numeric(AR22.model.coeffi.beta04.vector[p])
      
      # The same to RV_[t-4d]
      AR22.model.t.4.coeffi <- as.numeric(AR22.model.coeffi.beta05.vector[p])
      
      # The same to RV_[t-5d]
      AR22.model.t.5.coeffi <- as.numeric(AR22.model.coeffi.beta06.vector[p])
      
      # The same to RV_[t-6d]
      AR22.model.t.6.coeffi <- as.numeric(AR22.model.coeffi.beta07.vector[p])
      
      # The same to RV_[t-7d]
      AR22.model.t.7.coeffi <- as.numeric(AR22.model.coeffi.beta08.vector[p])
      
      # The same to RV_[t-8d]
      AR22.model.t.8.coeffi <- as.numeric(AR22.model.coeffi.beta09.vector[p])
      
      # The same to RV_[t-9d]
      AR22.model.t.9.coeffi <- as.numeric(AR22.model.coeffi.beta010.vector[p])
      
      # The same to RV_[t-10d]
      AR22.model.t.10.coeffi <- as.numeric(AR22.model.coeffi.beta011.vector[p])
      
      # The same to RV_[t-11d]
      AR22.model.t.11.coeffi <- as.numeric(AR22.model.coeffi.beta012.vector[p])
      
      # The same to RV_[t-12d]
      AR22.model.t.12.coeffi <- as.numeric(AR22.model.coeffi.beta013.vector[p])
      
      # The same to RV_[t-13d]
      AR22.model.t.13.coeffi <- as.numeric(AR22.model.coeffi.beta014.vector[p])
      
      # The same to RV_[t-14d]
      AR22.model.t.14.coeffi <- as.numeric(AR22.model.coeffi.beta015.vector[p])
      
      # The same to RV_[t-15d]
      AR22.model.t.15.coeffi <- as.numeric(AR22.model.coeffi.beta016.vector[p])
      
      # The same to RV_[t-16d]
      AR22.model.t.16.coeffi <- as.numeric(AR22.model.coeffi.beta017.vector[p])
      
      # The same to RV_[t-17d]
      AR22.model.t.17.coeffi <- as.numeric(AR22.model.coeffi.beta018.vector[p])
      
      # The same to RV_[t-18d]
      AR22.model.t.18.coeffi <- as.numeric(AR22.model.coeffi.beta019.vector[p])
      
      # The same to RV_[t-19d]
      AR22.model.t.19.coeffi <- as.numeric(AR22.model.coeffi.beta020.vector[p])
      
      # The same to RV_[t-20d]
      AR22.model.t.20.coeffi <- as.numeric(AR22.model.coeffi.beta021.vector[p])
      
      # The same to RV_[t-21d]
      AR22.model.t.21.coeffi <- as.numeric(AR22.model.coeffi.beta022.vector[p])
      
      
      break
    }
    
    ## Repeat the above steps (STEP3-> STEP4-> STEP3) repeatedly 
    ## until the coefficients corresponding to the AR(22) model PART of HAR-AR(22)-NN model 
    ## converge to a certain level. 
    
    
  }
  
  
  
  
########### STEP 5 : Fit the NN model lastly to the final.res data. 
  
  final.res <-(Rv.d.y2-AR22.model.t.coeffi*Rv.d.x1.2-AR22.model.t.1.coeffi*Rv.d.x1.2.1-AR22.model.t.2.coeffi*Rv.d.x1.2.2-AR22.model.t.3.coeffi*Rv.d.x1.2.3-AR22.model.t.4.coeffi*Rv.d.x1.2.4-AR22.model.t.5.coeffi*Rv.d.x1.2.5-AR22.model.t.6.coeffi*Rv.d.x1.2.6-AR22.model.t.7.coeffi*Rv.d.x1.2.7
               -AR22.model.t.8.coeffi*Rv.d.x1.2.8-AR22.model.t.9.coeffi*Rv.d.x1.2.9-AR22.model.t.10.coeffi*Rv.d.x1.2.10-AR22.model.t.11.coeffi*Rv.d.x1.2.11
               -AR22.model.t.12.coeffi*Rv.d.x1.2.12-AR22.model.t.13.coeffi*Rv.d.x1.2.13-AR22.model.t.14.coeffi*Rv.d.x1.2.14-AR22.model.t.15.coeffi*Rv.d.x1.2.15
               -AR22.model.t.16.coeffi*Rv.d.x1.2.16-AR22.model.t.17.coeffi*Rv.d.x1.2.17-AR22.model.t.18.coeffi*Rv.d.x1.2.18-AR22.model.t.19.coeffi*Rv.d.x1.2.19-AR22.model.t.20.coeffi*Rv.d.x1.2.20
               -AR22.model.t.21.coeffi*Rv.d.x1.2.21)
  
  ## 
  RVcombined.data3 <- cbind(final.res,Rv.d.x1.2,Rv.w.2,Rv.m.2)
  
  
  
  RVcombined.data3 <-as.data.frame(RVcombined.data3)
  
  
  
  NN.model.refitting.to.final.res <- neuralnet(final.res~Rv.d.x1.2+Rv.w.2+Rv.m.2,data=RVcombined.data3,
                                               hidden=5,linear.output=T)
  
  ################ sigmoid function values.
  
  a1 <- (1/(1+exp(-(NN.model.refitting.to.final.res$weights[[1]][[1]][,1][1] + NN.model.refitting.to.final.res$weights[[1]][[1]][,1][2]*Rv.d.x1[nrow(RV.data)-i] + NN.model.refitting.to.final.res$weights[[1]][[1]][,1][3]*Rv.w[nrow(RV.data)-i]+
                      NN.model.refitting.to.final.res$weights[[1]][[1]][,1][4]*Rv.m[nrow(RV.data)-i]))))
  
  a2 <- (1/(1+exp(-(NN.model.refitting.to.final.res$weights[[1]][[1]][,2][1] + NN.model.refitting.to.final.res$weights[[1]][[1]][,2][2]*Rv.d.x1[nrow(RV.data)-i] + NN.model.refitting.to.final.res$weights[[1]][[1]][,2][3]*Rv.w[nrow(RV.data)-i]+
                      NN.model.refitting.to.final.res$weights[[1]][[1]][,2][4]*Rv.m[nrow(RV.data)-i]))))
  
  a3 <- (1/(1+exp(-(NN.model.refitting.to.final.res$weights[[1]][[1]][,3][1] + NN.model.refitting.to.final.res$weights[[1]][[1]][,3][2]*Rv.d.x1[nrow(RV.data)-i] + NN.model.refitting.to.final.res$weights[[1]][[1]][,3][3]*Rv.w[nrow(RV.data)-i]+
                      NN.model.refitting.to.final.res$weights[[1]][[1]][,3][4]*Rv.m[nrow(RV.data)-i]))))
  
  a4 <- (1/(1+exp(-(NN.model.refitting.to.final.res$weights[[1]][[1]][,4][1] + NN.model.refitting.to.final.res$weights[[1]][[1]][,4][2]*Rv.d.x1[nrow(RV.data)-i] + NN.model.refitting.to.final.res$weights[[1]][[1]][,4][3]*Rv.w[nrow(RV.data)-i]+
                      NN.model.refitting.to.final.res$weights[[1]][[1]][,4][4]*Rv.m[nrow(RV.data)-i]))))
  
  a5 <- (1/(1+exp(-(NN.model.refitting.to.final.res$weights[[1]][[1]][,5][1] + NN.model.refitting.to.final.res$weights[[1]][[1]][,5][2]*Rv.d.x1[nrow(RV.data)-i] + NN.model.refitting.to.final.res$weights[[1]][[1]][,5][3]*Rv.w[nrow(RV.data)-i]+
                      NN.model.refitting.to.final.res$weights[[1]][[1]][,5][4]*Rv.m[nrow(RV.data)-i]))))
  
  
  ## All of the coefficients in the HAR-infty-NN model.
  
  
  coefficient.list <- list("Beta_01" = AR22.model.t.coeffi, "Beta_02" = AR22.model.t.1.coeffi, "Beta_03" = AR22.model.t.2.coeffi,
                           "Beta_04" = AR22.model.t.3.coeffi, "Beta_05" = AR22.model.t.4.coeffi, "Beta_06" = AR22.model.t.5.coeffi,
                           "Beta_07" = AR22.model.t.6.coeffi, "Beta_08" = AR22.model.t.7.coeffi, "Beta_09" = AR22.model.t.8.coeffi,
                           "Beta_010" = AR22.model.t.9.coeffi, "Beta_011" = AR22.model.t.10.coeffi, "Beta_012" = AR22.model.t.11.coeffi,
                           "Beta_013" = AR22.model.t.12.coeffi, "Beta_014" = AR22.model.t.13.coeffi, "Beta_015" = AR22.model.t.14.coeffi,
                           "Beta_016" = AR22.model.t.15.coeffi, "Beta_017" = AR22.model.t.16.coeffi, "Beta_018" = AR22.model.t.17.coeffi,
                           "Beta_019" = AR22.model.t.18.coeffi, "Beta_020" = AR22.model.t.19.coeffi, "Beta_021" = AR22.model.t.20.coeffi,
                           "Beta_022" = AR22.model.t.21.coeffi, "Beta_00" = NN.model.refitting.to.final.res$weights[[1]][[2]][[1]], 
                           
                           "beta_1" = NN.model.refitting.to.final.res$weights[[1]][[2]][[2]], 
                           "gamma_10" = NN.model.refitting.to.final.res$weights[[1]][[1]][,1][1], "gamma_1d" = NN.model.refitting.to.final.res$weights[[1]][[1]][,1][2], "gamma_1w" = NN.model.refitting.to.final.res$weights[[1]][[1]][,1][3], 
                           "gamma_1m" = NN.model.refitting.to.final.res$weights[[1]][[1]][,1][4],
                           
                           "beta_2" = NN.model.refitting.to.final.res$weights[[1]][[2]][[3]], 
                           "gamma_20" = NN.model.refitting.to.final.res$weights[[1]][[1]][,2][1], "gamma_2d" = NN.model.refitting.to.final.res$weights[[1]][[1]][,2][2], "gamma_2w" = NN.model.refitting.to.final.res$weights[[1]][[1]][,2][3], 
                           "gamma_2m" = NN.model.refitting.to.final.res$weights[[1]][[1]][,2][4], 
                           
                           "beta_3" = NN.model.refitting.to.final.res$weights[[1]][[2]][[4]], 
                           "gamma_30" = NN.model.refitting.to.final.res$weights[[1]][[1]][,3][1], "gamma_3d" = NN.model.refitting.to.final.res$weights[[1]][[1]][,3][2], "gamma_3w" = NN.model.refitting.to.final.res$weights[[1]][[1]][,3][3], 
                           "gamma_3m" = NN.model.refitting.to.final.res$weights[[1]][[1]][,3][4], 
                           
                           "beta_4" = NN.model.refitting.to.final.res$weights[[1]][[2]][[5]], 
                           "gamma_40" = NN.model.refitting.to.final.res$weights[[1]][[1]][,4][1], "gamma_4d" = NN.model.refitting.to.final.res$weights[[1]][[1]][,4][2], "gamma_4w" = NN.model.refitting.to.final.res$weights[[1]][[1]][,4][3], 
                           "gamma_4m" = NN.model.refitting.to.final.res$weights[[1]][[1]][,4][4], 
                           
                           "beta_5" = NN.model.refitting.to.final.res$weights[[1]][[2]][[6]], 
                           "gamma_50" = NN.model.refitting.to.final.res$weights[[1]][[1]][,5][1], "gamma_5d" = NN.model.refitting.to.final.res$weights[[1]][[1]][,5][2], "gamma_5w" = NN.model.refitting.to.final.res$weights[[1]][[1]][,5][3], 
                           "gamma_5m" = NN.model.refitting.to.final.res$weights[[1]][[1]][,5][4])
  
                           
  #### Predicted value for RV_[T+1d]^(d), where T = nrow(RV.data)-i.
  predicted.values.for.RV.Tplus1d.d  <- AR22.model.t.coeffi*Rv.d.x1[nrow(RV.data)-i] + AR22.model.t.1.coeffi*Rv.d.x1[nrow(RV.data)-(i+1)]+
    AR22.model.t.2.coeffi*Rv.d.x1[nrow(RV.data)-(i+2)]+AR22.model.t.3.coeffi*Rv.d.x1[nrow(RV.data)-(i+3)]+AR22.model.t.4.coeffi*Rv.d.x1[nrow(RV.data)-(i+4)]+
    AR22.model.t.5.coeffi*Rv.d.x1[nrow(RV.data)-(i+5)]+AR22.model.t.6.coeffi*Rv.d.x1[nrow(RV.data)-(i+6)]+
    AR22.model.t.7.coeffi*Rv.d.x1[nrow(RV.data)-(i+7)]+AR22.model.t.8.coeffi*Rv.d.x1[nrow(RV.data)-(i+8)]+AR22.model.t.9.coeffi*Rv.d.x1[nrow(RV.data)-(i+9)]
  +AR22.model.t.10.coeffi*Rv.d.x1[nrow(RV.data)-(i+10)]+AR22.model.t.11.coeffi*Rv.d.x1[nrow(RV.data)-(i+11)]+AR22.model.t.12.coeffi*Rv.d.x1[nrow(RV.data)-(i+12)]
  +AR22.model.t.13.coeffi*Rv.d.x1[nrow(RV.data)-(i+13)]+AR22.model.t.14.coeffi*Rv.d.x1[nrow(RV.data)-(i+14)]+AR22.model.t.15.coeffi*Rv.d.x1[nrow(RV.data)-(i+15)]
  +AR22.model.t.16.coeffi*Rv.d.x1[nrow(RV.data)-(i+16)]+AR22.model.t.17.coeffi*Rv.d.x1[nrow(RV.data)-(i+17)]+AR22.model.t.18.coeffi*Rv.d.x1[nrow(RV.data)-(i+18)]
  +AR22.model.t.19.coeffi*Rv.d.x1[nrow(RV.data)-(i+19)]+AR22.model.t.20.coeffi*Rv.d.x1[nrow(RV.data)-(i+20)]
  +AR22.model.t.21.coeffi*Rv.d.x1[nrow(RV.data)-(i+21)]+NN.model.refitting.to.final.res$weights[[1]][[2]][[1]]+NN.model.refitting.to.final.res$weights[[1]][[2]][[2]]*a1+NN.model.refitting.to.final.res$weights[[1]][[2]][[3]]*a2+
    NN.model.refitting.to.final.res$weights[[1]][[2]][[4]]*a3+NN.model.refitting.to.final.res$weights[[1]][[2]][[5]]*a4+NN.model.refitting.to.final.res$weights[[1]][[2]][[6]]*a5
  
  
  if(printcoeffi=="YES"){
    print(coefficient.list)
  }
  
  else if(printcoeffi=="NO"){
    print(predicted.values.for.RV.Tplus1d.d)
  }
  
  else if(printcoeffi=="BOTH"){
    print(coefficient.list)
    print(predicted.values.for.RV.Tplus1d.d)
  }
  
  
  else
    print("You might have a typo error in 'printcoeffi' argument")
  
}


# HAR_AR22_NN.sig(KOSPI.0615.data.RV,100,200,"YES")



#####################################

HAR_AR22_NN.tanh <- function(RV.data,i,r,printcoeffi){
  
  Rv.d.y <- as.numeric(as.vector(RV.data[c(2:((nrow(RV.data)-i)+1)),2])) ## from 2 (daily time point) ~ (nrow(RV.data)-100)+1 (daily daily time point). 
  
  Rv.d.x1 <- as.numeric(as.vector(RV.data[c(1:(nrow(RV.data)-i)),2])) ## from 1 (daily time point) ~ (nrow(RV.data)-100) (daily daily time point). 
  
  ## Data handling for AR(22) model fitting
  
  Rv.d.y2 <- Rv.d.y[23:(nrow(RV.data)-i)]  # (t+1d) daily time point series data
  
  Rv.d.x1.2 <- Rv.d.x1[23:(nrow(RV.data)-i)] # (t) daily time point series data
  
  
  ## Weekly, monthly RV.
  
  Rv.w <- numeric((nrow(RV.data)-i))
  for(j in 23:(nrow(RV.data)-i)){
    Rv.w[j] <- (Rv.d.x1[j]+Rv.d.x1[j-1]+Rv.d.x1[j-2]+Rv.d.x1[j-3]+Rv.d.x1[j-4])/5
  }
  
  
  Rv.m <- numeric((nrow(RV.data)-i)) 
  for(q in 23:(nrow(RV.data)-i)){
    Rv.m[q] <- (Rv.d.x1[q]+Rv.d.x1[q-1]+Rv.d.x1[q-2]+Rv.d.x1[q-3]
                +Rv.d.x1[q-4]+Rv.d.x1[q-5]+Rv.d.x1[q-6]+Rv.d.x1[q-7]
                +Rv.d.x1[q-8]+Rv.d.x1[q-9]+Rv.d.x1[q-10]+Rv.d.x1[q-11]
                +Rv.d.x1[q-12]+Rv.d.x1[q-13]+Rv.d.x1[q-14]+Rv.d.x1[q-15]
                +Rv.d.x1[q-16]+Rv.d.x1[q-17]+Rv.d.x1[q-18]+Rv.d.x1[q-19]
                +Rv.d.x1[q-20]+Rv.d.x1[q-21])/22
  }
  
  Rv.w.2 <- Rv.w[23:i]
  Rv.m.2 <- Rv.m[23:i]
  
  
########### STEP 1 : Initial AR(22) model fitting to data set. 
  
  initial.AR22.model.fit <- arima(Rv.d.y2,order=c(22,0,0),method="CSS") 
  
  initial.AR22.model.coeffi <- initial.AR22.model.fit$coef
  
  initial.AR22.model.coeffi <- as.vector(initial.AR22.model.coeffi)
  
  # (t-1d) daily time point series data
  Rv.d.x1.2.1 <- c(Rv.d.x1[22],Rv.d.x1.2[-length(Rv.d.x1.2)])
  
  # (t-2d) daily time point series data
  Rv.d.x1.2.2 <- c(Rv.d.x1[21],Rv.d.x1[22],Rv.d.x1.2[-(length(Rv.d.x1.2)-1):-length(Rv.d.x1.2)])
  
  # (t-3d) daily time point series data
  Rv.d.x1.2.3 <- c(Rv.d.x1[20],Rv.d.x1[21],Rv.d.x1[22],Rv.d.x1.2[-(length(Rv.d.x1.2)-2):-length(Rv.d.x1.2)])
  
  # (t-4d) daily time point series data
  Rv.d.x1.2.4 <- c(Rv.d.x1[19],Rv.d.x1[20],Rv.d.x1[21],Rv.d.x1[22],Rv.d.x1.2[-(length(Rv.d.x1.2)-3):-length(Rv.d.x1.2)])
  
  # (t-5d) daily time point series data
  Rv.d.x1.2.5 <- c(Rv.d.x1[18],Rv.d.x1[19],Rv.d.x1[20],Rv.d.x1[21],Rv.d.x1[22],Rv.d.x1.2[-(length(Rv.d.x1.2)-4):-length(Rv.d.x1.2)])
  
  # (t-6d) daily time point series data
  Rv.d.x1.2.6 <- c(Rv.d.x1[17],Rv.d.x1[18],Rv.d.x1[19],Rv.d.x1[20],Rv.d.x1[21],Rv.d.x1[22],Rv.d.x1.2[-(length(Rv.d.x1.2)-5):-length(Rv.d.x1.2)])
  
  # (t-7d) daily time point series data
  Rv.d.x1.2.7 <- c(Rv.d.x1[16],Rv.d.x1[17],Rv.d.x1[18],Rv.d.x1[19],Rv.d.x1[20],Rv.d.x1[21],Rv.d.x1[22],Rv.d.x1.2[-(length(Rv.d.x1.2)-6):-length(Rv.d.x1.2)])
  
  # (t-8d) daily time point series data
  Rv.d.x1.2.8 <- c(Rv.d.x1[15],Rv.d.x1[16],Rv.d.x1[17],Rv.d.x1[18],Rv.d.x1[19],Rv.d.x1[20],Rv.d.x1[21],Rv.d.x1[22],Rv.d.x1.2[-(length(Rv.d.x1.2)-7):-length(Rv.d.x1.2)])
  
  # (t-9d) daily time point series data
  Rv.d.x1.2.9 <- c(Rv.d.x1[14],Rv.d.x1[15],Rv.d.x1[16],Rv.d.x1[17],Rv.d.x1[18],Rv.d.x1[19],Rv.d.x1[20],Rv.d.x1[21],Rv.d.x1[22],Rv.d.x1.2[-(length(Rv.d.x1.2)-8):-length(Rv.d.x1.2)])
  
  # (t-10d) daily time point series data
  Rv.d.x1.2.10 <- c(Rv.d.x1[13],Rv.d.x1[14],Rv.d.x1[15],Rv.d.x1[16],Rv.d.x1[17],Rv.d.x1[18],Rv.d.x1[19],Rv.d.x1[20],Rv.d.x1[21],Rv.d.x1[22],Rv.d.x1.2[-(length(Rv.d.x1.2)-9):-length(Rv.d.x1.2)])
  
  # (t-11d) daily time point series data
  Rv.d.x1.2.11 <- c(Rv.d.x1[12],Rv.d.x1[13],Rv.d.x1[14],Rv.d.x1[15],Rv.d.x1[16],Rv.d.x1[17],Rv.d.x1[18],Rv.d.x1[19],Rv.d.x1[20],Rv.d.x1[21],Rv.d.x1[22],Rv.d.x1.2[-(length(Rv.d.x1.2)-10):-length(Rv.d.x1.2)])
  
  # (t-12d) daily time point series data
  Rv.d.x1.2.12 <- c(Rv.d.x1[11],Rv.d.x1[12],Rv.d.x1[13],Rv.d.x1[14],Rv.d.x1[15],Rv.d.x1[16],Rv.d.x1[17],Rv.d.x1[18],Rv.d.x1[19],Rv.d.x1[20],Rv.d.x1[21],Rv.d.x1[22],Rv.d.x1.2[-(length(Rv.d.x1.2)-11):-length(Rv.d.x1.2)])
  
  # (t-13d) daily time point series data
  Rv.d.x1.2.13 <- c(Rv.d.x1[10],Rv.d.x1[11],Rv.d.x1[12],Rv.d.x1[13],Rv.d.x1[14],Rv.d.x1[15],Rv.d.x1[16],Rv.d.x1[17],Rv.d.x1[18],Rv.d.x1[19],Rv.d.x1[20],Rv.d.x1[21],Rv.d.x1[22],Rv.d.x1.2[-(length(Rv.d.x1.2)-12):-length(Rv.d.x1.2)])
  
  # (t-14d) daily time point series data
  Rv.d.x1.2.14 <- c(Rv.d.x1[9],Rv.d.x1[10],Rv.d.x1[11],Rv.d.x1[12],Rv.d.x1[13],Rv.d.x1[14],Rv.d.x1[15],Rv.d.x1[16],Rv.d.x1[17],Rv.d.x1[18],Rv.d.x1[19],Rv.d.x1[20],Rv.d.x1[21],Rv.d.x1[22],Rv.d.x1.2[-(length(Rv.d.x1.2)-13):-length(Rv.d.x1.2)])
  
  # (t-15d) daily time point series data
  Rv.d.x1.2.15 <- c(Rv.d.x1[8],Rv.d.x1[9],Rv.d.x1[10],Rv.d.x1[11],Rv.d.x1[12],Rv.d.x1[13],Rv.d.x1[14],Rv.d.x1[15],Rv.d.x1[16],Rv.d.x1[17],Rv.d.x1[18],Rv.d.x1[19],Rv.d.x1[20],Rv.d.x1[21],Rv.d.x1[22],Rv.d.x1.2[-(length(Rv.d.x1.2)-14):-length(Rv.d.x1.2)])
  
  # (t-16d) daily time point series data
  Rv.d.x1.2.16 <- c(Rv.d.x1[7],Rv.d.x1[8],Rv.d.x1[9],Rv.d.x1[10],Rv.d.x1[11],Rv.d.x1[12],Rv.d.x1[13],Rv.d.x1[14],Rv.d.x1[15],Rv.d.x1[16],Rv.d.x1[17],Rv.d.x1[18],Rv.d.x1[19],Rv.d.x1[20],Rv.d.x1[21],Rv.d.x1[22],Rv.d.x1.2[-(length(Rv.d.x1.2)-15):-length(Rv.d.x1.2)])
  
  # (t-17d) daily time point series data
  Rv.d.x1.2.17 <- c(Rv.d.x1[6],Rv.d.x1[7],Rv.d.x1[8],Rv.d.x1[9],Rv.d.x1[10],Rv.d.x1[11],Rv.d.x1[12],Rv.d.x1[13],Rv.d.x1[14],Rv.d.x1[15],Rv.d.x1[16],Rv.d.x1[17],Rv.d.x1[18],Rv.d.x1[19],Rv.d.x1[20],Rv.d.x1[21],Rv.d.x1[22],Rv.d.x1.2[-(length(Rv.d.x1.2)-16):-length(Rv.d.x1.2)])
  
  # (t-18d) daily time point series data
  Rv.d.x1.2.18 <- c(Rv.d.x1[5],Rv.d.x1[6],Rv.d.x1[7],Rv.d.x1[8],Rv.d.x1[9],Rv.d.x1[10],Rv.d.x1[11],Rv.d.x1[12],Rv.d.x1[13],Rv.d.x1[14],Rv.d.x1[15],Rv.d.x1[16],Rv.d.x1[17],Rv.d.x1[18],Rv.d.x1[19],Rv.d.x1[20],Rv.d.x1[21],Rv.d.x1[22],Rv.d.x1.2[-(length(Rv.d.x1.2)-17):-length(Rv.d.x1.2)])
  
  # (t-19d) daily time point series data
  Rv.d.x1.2.19 <- c(Rv.d.x1[4],Rv.d.x1[5],Rv.d.x1[6],Rv.d.x1[7],Rv.d.x1[8],Rv.d.x1[9],Rv.d.x1[10],Rv.d.x1[11],Rv.d.x1[12],Rv.d.x1[13],Rv.d.x1[14],Rv.d.x1[15],Rv.d.x1[16],Rv.d.x1[17],Rv.d.x1[18],Rv.d.x1[19],Rv.d.x1[20],Rv.d.x1[21],Rv.d.x1[22],Rv.d.x1.2[-(length(Rv.d.x1.2)-18):-length(Rv.d.x1.2)])
  
  # (t-20d) daily time point series data
  Rv.d.x1.2.20 <- c(Rv.d.x1[3],Rv.d.x1[4],Rv.d.x1[5],Rv.d.x1[6],Rv.d.x1[7],Rv.d.x1[8],Rv.d.x1[9],Rv.d.x1[10],Rv.d.x1[11],Rv.d.x1[12],Rv.d.x1[13],Rv.d.x1[14],Rv.d.x1[15],Rv.d.x1[16],Rv.d.x1[17],Rv.d.x1[18],Rv.d.x1[19],Rv.d.x1[20],Rv.d.x1[21],Rv.d.x1[22],Rv.d.x1.2[-(length(Rv.d.x1.2)-19):-length(Rv.d.x1.2)])
  
  # (t-21d) daily time point series data
  Rv.d.x1.2.21 <- c(Rv.d.x1[2],Rv.d.x1[3],Rv.d.x1[4],Rv.d.x1[5],Rv.d.x1[6],Rv.d.x1[7],Rv.d.x1[8],Rv.d.x1[9],Rv.d.x1[10],Rv.d.x1[11],Rv.d.x1[12],Rv.d.x1[13],Rv.d.x1[14],Rv.d.x1[15],Rv.d.x1[16],Rv.d.x1[17],Rv.d.x1[18],Rv.d.x1[19],Rv.d.x1[20],Rv.d.x1[21],Rv.d.x1[22],Rv.d.x1.2[-(length(Rv.d.x1.2)-20):-length(Rv.d.x1.2)])
  
  
  
  ## initial residual
  
  initial.res <-(Rv.d.y2-initial.AR22.model.coeffi[1]*Rv.d.x1.2-initial.AR22.model.coeffi[2]*Rv.d.x1.2.1-initial.AR22.model.coeffi[3]*Rv.d.x1.2.2
                 -initial.AR22.model.coeffi[4]*Rv.d.x1.2.3-initial.AR22.model.coeffi[5]*Rv.d.x1.2.4-initial.AR22.model.coeffi[6]*Rv.d.x1.2.5-initial.AR22.model.coeffi[7]*Rv.d.x1.2.6
                 -initial.AR22.model.coeffi[8]*Rv.d.x1.2.7-initial.AR22.model.coeffi[9]*Rv.d.x1.2.8-initial.AR22.model.coeffi[10]*Rv.d.x1.2.9-initial.AR22.model.coeffi[11]*Rv.d.x1.2.10-initial.AR22.model.coeffi[12]*Rv.d.x1.2.11
                 -initial.AR22.model.coeffi[13]*Rv.d.x1.2.12-initial.AR22.model.coeffi[14]*Rv.d.x1.2.13-initial.AR22.model.coeffi[15]*Rv.d.x1.2.14-initial.AR22.model.coeffi[16]*Rv.d.x1.2.15
                 -initial.AR22.model.coeffi[17]*Rv.d.x1.2.16-initial.AR22.model.coeffi[18]*Rv.d.x1.2.17-initial.AR22.model.coeffi[19]*Rv.d.x1.2.18-initial.AR22.model.coeffi[20]*Rv.d.x1.2.19-initial.AR22.model.coeffi[21]*Rv.d.x1.2.20
                 -initial.AR22.model.coeffi[22]*Rv.d.x1.2.21)
  
  
  RVcombined.data <- cbind(initial.res,Rv.d.x1.2,Rv.w.2,Rv.m.2)
  
########### STEP 2 : Initial NN fitting to initial residual set. 
  
  ## NN model fitting to initial res data.
  
  NN.model.fitting.to.initial.res <- neuralnet(initial.res~Rv.d.x1.2+Rv.w.2+Rv.m.2,data=RVcombined.data,
                                               hidden=10,linear.output=T,act.fct = "tanh")
  
  
  ## The AR(22) model coefficient value corresponding to RV_[t] (numeric vector to store the updated value)
  AR22.model.coeffi.beta01.vector <- numeric(r)
  u.AR22.model.coeffi.beta01.vector <- numeric(r)
  
  ## The AR(22) model coefficient value corresponding to RV_[t-1d]
  AR22.model.coeffi.beta02.vector <- numeric(r)
  u.AR22.model.coeffi.beta02.vector <- numeric(r)
  
  ## The same to RV_[t-2d]
  AR22.model.coeffi.beta03.vector <- numeric(r)
  u.AR22.model.coeffi.beta03.vector <- numeric(r)
  
  ## The same to RV_[t-3d]
  AR22.model.coeffi.beta04.vector <- numeric(r)
  u.AR22.model.coeffi.beta04.vector <- numeric(r)
  
  ## The same to RV_[t-4d]
  AR22.model.coeffi.beta05.vector <- numeric(r)
  u.AR22.model.coeffi.beta05.vector <- numeric(r)
  
  ## The same to RV_[t-5d]
  AR22.model.coeffi.beta06.vector <- numeric(r)
  u.AR22.model.coeffi.beta06.vector <- numeric(r)
  
  ## The same to RV_[t-6d]
  AR22.model.coeffi.beta07.vector <- numeric(r)
  u.AR22.model.coeffi.beta07.vector <- numeric(r)
  
  ## The same to RV_[t-7d]
  AR22.model.coeffi.beta08.vector <- numeric(r)
  u.AR22.model.coeffi.beta08.vector <- numeric(r)
  
  ## The same to RV_[t-8d]
  AR22.model.coeffi.beta09.vector <- numeric(r)
  u.AR22.model.coeffi.beta09.vector <- numeric(r)
  
  ## The same to RV_[t-9d]
  AR22.model.coeffi.beta010.vector <- numeric(r)
  u.AR22.model.coeffi.beta010.vector <- numeric(r)
  
  ## The same to RV_[t-10d]
  AR22.model.coeffi.beta011.vector <- numeric(r)
  u.AR22.model.coeffi.beta011.vector <- numeric(r)
  
  ## The same to RV_[t-11d]
  AR22.model.coeffi.beta012.vector <- numeric(r)
  u.AR22.model.coeffi.beta012.vector <- numeric(r)
  
  ## The same to RV_[t-12d]
  AR22.model.coeffi.beta013.vector <- numeric(r)
  u.AR22.model.coeffi.beta013.vector <- numeric(r)
  
  ## The same to RV_[t-13d]
  AR22.model.coeffi.beta014.vector <- numeric(r)
  u.AR22.model.coeffi.beta014.vector <- numeric(r)
  
  ## The same to RV_[t-14d]
  AR22.model.coeffi.beta015.vector <- numeric(r)
  u.AR22.model.coeffi.beta015.vector <- numeric(r)
  
  ## The same to RV_[t-15d]
  AR22.model.coeffi.beta016.vector <- numeric(r)
  u.AR22.model.coeffi.beta016.vector <- numeric(r)
  
  ## The same to RV_[t-16d]
  AR22.model.coeffi.beta017.vector <- numeric(r)
  u.AR22.model.coeffi.beta017.vector <- numeric(r)
  
  ## The same to RV_[t-17d]
  AR22.model.coeffi.beta018.vector <- numeric(r)
  u.AR22.model.coeffi.beta018.vector <- numeric(r)
  
  ## The same to RV_[t-18d]
  AR22.model.coeffi.beta019.vector <- numeric(r)
  u.AR22.model.coeffi.beta019.vector <- numeric(r)
  
  ## The same to RV_[t-19d]
  AR22.model.coeffi.beta020.vector <- numeric(r)
  u.AR22.model.coeffi.beta020.vector <- numeric(r)
  
  ## The same to RV_[t-20d]
  AR22.model.coeffi.beta021.vector <- numeric(r)
  u.AR22.model.coeffi.beta021.vector <- numeric(r)
  
  ## The same to RV_[t-21d]
  AR22.model.coeffi.beta022.vector <- numeric(r)
  u.AR22.model.coeffi.beta022.vector <- numeric(r)
  
  
  
  
  for(p in 1:r){
    
    
########### STEP 3 : refit.AR22.model to ( Rv_[t+1d](d) - fitted.value of 'NN.model.fitting.to.initial.res' )       
    
    refit.AR22.model <- arima((Rv.d.y2 - NN.model.fitting.to.initial.res$net.result[[1]]),order=c(22,0,0),method="CSS")
    
    ### coefficient
    
    refit.AR22.model.coeffi <-(refit.AR22.model$coef)
    refit.AR22.model.coeffi <- as.vector(refit.AR22.model.coeffi)
    refit.AR22.model.coeffi ## 
    
    
    ## store AR(22) Coefficients obtained from STEP 3
    
    
    AR22.model.coeffi.beta01.vector[p] <- refit.AR22.model.coeffi[1]
    
    AR22.model.coeffi.beta02.vector[p] <- refit.AR22.model.coeffi[2]
    
    AR22.model.coeffi.beta03.vector[p] <- refit.AR22.model.coeffi[3]
    
    AR22.model.coeffi.beta04.vector[p] <- refit.AR22.model.coeffi[4]
    
    AR22.model.coeffi.beta05.vector[p] <- refit.AR22.model.coeffi[5]
    
    AR22.model.coeffi.beta06.vector[p] <- refit.AR22.model.coeffi[6]
    
    AR22.model.coeffi.beta07.vector[p] <- refit.AR22.model.coeffi[7]
    
    AR22.model.coeffi.beta08.vector[p] <- refit.AR22.model.coeffi[8]
    
    AR22.model.coeffi.beta09.vector[p] <- refit.AR22.model.coeffi[9]
    
    AR22.model.coeffi.beta010.vector[p] <- refit.AR22.model.coeffi[10]
    
    AR22.model.coeffi.beta011.vector[p] <- refit.AR22.model.coeffi[11]
    
    AR22.model.coeffi.beta012.vector[p] <- refit.AR22.model.coeffi[12]
    
    AR22.model.coeffi.beta013.vector[p] <- refit.AR22.model.coeffi[13]
    
    AR22.model.coeffi.beta014.vector[p] <- refit.AR22.model.coeffi[14]
    
    AR22.model.coeffi.beta015.vector[p] <- refit.AR22.model.coeffi[15]
    
    AR22.model.coeffi.beta016.vector[p] <- refit.AR22.model.coeffi[16]
    
    AR22.model.coeffi.beta017.vector[p] <- refit.AR22.model.coeffi[17]
    
    AR22.model.coeffi.beta018.vector[p] <- refit.AR22.model.coeffi[18]
    
    AR22.model.coeffi.beta019.vector[p] <- refit.AR22.model.coeffi[19]
    
    AR22.model.coeffi.beta020.vector[p] <- refit.AR22.model.coeffi[20]
    
    AR22.model.coeffi.beta021.vector[p] <- refit.AR22.model.coeffi[21]
    
    AR22.model.coeffi.beta022.vector[p] <- refit.AR22.model.coeffi[22]
    
    
########### STEP 4 : Fit NN again to second.res data.
    
    
    # second.res
    second.res <-(Rv.d.y2-AR22.model.coeffi.beta01.vector[p]*Rv.d.x1.2-AR22.model.coeffi.beta02.vector[p]*Rv.d.x1.2.1-AR22.model.coeffi.beta03.vector[p]*Rv.d.x1.2.2-AR22.model.coeffi.beta04.vector[p]*Rv.d.x1.2.3-AR22.model.coeffi.beta05.vector[p]*Rv.d.x1.2.4-AR22.model.coeffi.beta06.vector[p]*Rv.d.x1.2.5-AR22.model.coeffi.beta07.vector[p]*Rv.d.x1.2.6-AR22.model.coeffi.beta08.vector[p]*Rv.d.x1.2.7
                  -AR22.model.coeffi.beta09.vector[p]*Rv.d.x1.2.8-AR22.model.coeffi.beta010.vector[p]*Rv.d.x1.2.9-AR22.model.coeffi.beta011.vector[p]*Rv.d.x1.2.10-AR22.model.coeffi.beta012.vector[p]*Rv.d.x1.2.11
                  -AR22.model.coeffi.beta013.vector[p]*Rv.d.x1.2.12-AR22.model.coeffi.beta014.vector[p]*Rv.d.x1.2.13-AR22.model.coeffi.beta015.vector[p]*Rv.d.x1.2.14-AR22.model.coeffi.beta016.vector[p]*Rv.d.x1.2.15
                  -AR22.model.coeffi.beta017.vector[p]*Rv.d.x1.2.16-AR22.model.coeffi.beta018.vector[p]*Rv.d.x1.2.17-AR22.model.coeffi.beta019.vector[p]*Rv.d.x1.2.18-AR22.model.coeffi.beta020.vector[p]*Rv.d.x1.2.19-AR22.model.coeffi.beta021.vector[p]*Rv.d.x1.2.20
                  -AR22.model.coeffi.beta022.vector[p]*Rv.d.x1.2.21)
    
    
    RVcombined.data2 <- cbind(second.res,Rv.d.x1.2,Rv.w.2,Rv.m.2)
    
    
    ## re-fit NN model to second.res data.
    refit.NN.model.to.second.res <- neuralnet(second.res~Rv.d.x1.2+Rv.w.2+Rv.m.2,data=RVcombined.data2,
                                              hidden=10,linear.output=T,act.fct = "tanh")
    
    
    
############ Return to the STEP 3 : rerefit.AR22.model to ( Rv_[t+1d](d) - fitted.value of 'refit.NN.model.to.second.res' ) 
    
    rere.AR22.model.fit <- arima((Rv.d.y2 - refit.NN.model.to.second.res$net.result[[1]]),order=c(22,0,0),method="CSS")
    
    ### coefficient
    
    rere.AR22.model.coeffi <-(rere.AR22.model.fit$coef)
    rere.AR22.model.coeffi <- as.vector(rere.AR22.model.coeffi)
    
    ## store updated AR(22) Coefficients obtained from 'Return to the STEP 3'.
    
    u.AR22.model.coeffi.beta01.vector[p] <- rere.AR22.model.coeffi[1] 
    
    u.AR22.model.coeffi.beta02.vector[p] <- rere.AR22.model.coeffi[2] 
    
    u.AR22.model.coeffi.beta03.vector[p] <- rere.AR22.model.coeffi[3] 
    
    u.AR22.model.coeffi.beta04.vector[p] <- rere.AR22.model.coeffi[4] 
    
    u.AR22.model.coeffi.beta05.vector[p] <- rere.AR22.model.coeffi[5] 
    
    u.AR22.model.coeffi.beta06.vector[p] <- rere.AR22.model.coeffi[6] 
    
    u.AR22.model.coeffi.beta07.vector[p] <- rere.AR22.model.coeffi[7] 
    
    u.AR22.model.coeffi.beta08.vector[p] <- rere.AR22.model.coeffi[8] 
    
    u.AR22.model.coeffi.beta09.vector[p] <- rere.AR22.model.coeffi[9] 
    
    u.AR22.model.coeffi.beta010.vector[p] <- rere.AR22.model.coeffi[10] 
    
    u.AR22.model.coeffi.beta011.vector[p] <- rere.AR22.model.coeffi[11] 
    
    u.AR22.model.coeffi.beta012.vector[p] <- rere.AR22.model.coeffi[12] 
    
    u.AR22.model.coeffi.beta013.vector[p] <- rere.AR22.model.coeffi[13] 
    
    u.AR22.model.coeffi.beta014.vector[p] <- rere.AR22.model.coeffi[14] 
    
    u.AR22.model.coeffi.beta015.vector[p] <- rere.AR22.model.coeffi[15] 
    
    u.AR22.model.coeffi.beta016.vector[p] <- rere.AR22.model.coeffi[16] 
    
    u.AR22.model.coeffi.beta017.vector[p] <- rere.AR22.model.coeffi[17] 
    
    u.AR22.model.coeffi.beta018.vector[p] <- rere.AR22.model.coeffi[18] 
    
    u.AR22.model.coeffi.beta019.vector[p] <- rere.AR22.model.coeffi[19] 
    
    u.AR22.model.coeffi.beta020.vector[p] <- rere.AR22.model.coeffi[20] 
    
    u.AR22.model.coeffi.beta021.vector[p] <- rere.AR22.model.coeffi[21] 
    
    u.AR22.model.coeffi.beta022.vector[p] <- rere.AR22.model.coeffi[22] 
    
    
    ## 
    
    diff1=abs(AR22.model.coeffi.beta01.vector[p]-u.AR22.model.coeffi.beta01.vector[p])
    diff2=abs(AR22.model.coeffi.beta02.vector[p]-u.AR22.model.coeffi.beta02.vector[p])
    diff3=abs(AR22.model.coeffi.beta03.vector[p]-u.AR22.model.coeffi.beta03.vector[p])
    diff4=abs(AR22.model.coeffi.beta04.vector[p]-u.AR22.model.coeffi.beta04.vector[p])
    diff5=abs(AR22.model.coeffi.beta05.vector[p]-u.AR22.model.coeffi.beta05.vector[p])
    diff6=abs(AR22.model.coeffi.beta06.vector[p]-u.AR22.model.coeffi.beta06.vector[p])
    diff7=abs(AR22.model.coeffi.beta07.vector[p]-u.AR22.model.coeffi.beta07.vector[p])
    diff8=abs(AR22.model.coeffi.beta08.vector[p]-u.AR22.model.coeffi.beta08.vector[p])
    diff9=abs(AR22.model.coeffi.beta09.vector[p]-u.AR22.model.coeffi.beta09.vector[p])
    diff10=abs(AR22.model.coeffi.beta010.vector[p]-u.AR22.model.coeffi.beta010.vector[p])
    diff11=abs(AR22.model.coeffi.beta011.vector[p]-u.AR22.model.coeffi.beta011.vector[p])
    diff12=abs(AR22.model.coeffi.beta012.vector[p]-u.AR22.model.coeffi.beta012.vector[p])
    diff13=abs(AR22.model.coeffi.beta013.vector[p]-u.AR22.model.coeffi.beta013.vector[p])
    diff14=abs(AR22.model.coeffi.beta014.vector[p]-u.AR22.model.coeffi.beta014.vector[p])
    diff15=abs(AR22.model.coeffi.beta015.vector[p]-u.AR22.model.coeffi.beta015.vector[p])
    diff16=abs(AR22.model.coeffi.beta016.vector[p]-u.AR22.model.coeffi.beta016.vector[p])
    diff17=abs(AR22.model.coeffi.beta017.vector[p]-u.AR22.model.coeffi.beta017.vector[p])
    diff18=abs(AR22.model.coeffi.beta018.vector[p]-u.AR22.model.coeffi.beta018.vector[p])
    diff19=abs(AR22.model.coeffi.beta019.vector[p]-u.AR22.model.coeffi.beta019.vector[p])
    diff20=abs(AR22.model.coeffi.beta020.vector[p]-u.AR22.model.coeffi.beta020.vector[p])
    diff21=abs(AR22.model.coeffi.beta021.vector[p]-u.AR22.model.coeffi.beta021.vector[p])
    diff22=abs(AR22.model.coeffi.beta022.vector[p]-u.AR22.model.coeffi.beta022.vector[p])
    
    
    
    
    if( sum(c(diff1,diff2,diff3,diff4,diff5,diff6,diff7,diff8,diff9,diff10,diff11,diff12,diff13,diff14,diff15,diff16,diff17,diff18,diff19,diff20,diff21,diff22)<0.05)==22 )
    {
      
      # convergent coefficient corresponding to RV_[t] in AR(22) model part of HAR-infty-NN model. 
      AR22.model.t.coeffi <- as.numeric(AR22.model.coeffi.beta01.vector[p])
      
      # The same to RV_[t-1d]
      AR22.model.t.1.coeffi <- as.numeric(AR22.model.coeffi.beta02.vector[p])
      
      # The same to RV_[t-2d]
      AR22.model.t.2.coeffi <- as.numeric(AR22.model.coeffi.beta03.vector[p])
      
      # The same to RV_[t-3d]
      AR22.model.t.3.coeffi <- as.numeric(AR22.model.coeffi.beta04.vector[p])
      
      # The same to RV_[t-4d]
      AR22.model.t.4.coeffi <- as.numeric(AR22.model.coeffi.beta05.vector[p])
      
      # The same to RV_[t-5d]
      AR22.model.t.5.coeffi <- as.numeric(AR22.model.coeffi.beta06.vector[p])
      
      # The same to RV_[t-6d]
      AR22.model.t.6.coeffi <- as.numeric(AR22.model.coeffi.beta07.vector[p])
      
      # The same to RV_[t-7d]
      AR22.model.t.7.coeffi <- as.numeric(AR22.model.coeffi.beta08.vector[p])
      
      # The same to RV_[t-8d]
      AR22.model.t.8.coeffi <- as.numeric(AR22.model.coeffi.beta09.vector[p])
      
      # The same to RV_[t-9d]
      AR22.model.t.9.coeffi <- as.numeric(AR22.model.coeffi.beta010.vector[p])
      
      # The same to RV_[t-10d]
      AR22.model.t.10.coeffi <- as.numeric(AR22.model.coeffi.beta011.vector[p])
      
      # The same to RV_[t-11d]
      AR22.model.t.11.coeffi <- as.numeric(AR22.model.coeffi.beta012.vector[p])
      
      # The same to RV_[t-12d]
      AR22.model.t.12.coeffi <- as.numeric(AR22.model.coeffi.beta013.vector[p])
      
      # The same to RV_[t-13d]
      AR22.model.t.13.coeffi <- as.numeric(AR22.model.coeffi.beta014.vector[p])
      
      # The same to RV_[t-14d]
      AR22.model.t.14.coeffi <- as.numeric(AR22.model.coeffi.beta015.vector[p])
      
      # The same to RV_[t-15d]
      AR22.model.t.15.coeffi <- as.numeric(AR22.model.coeffi.beta016.vector[p])
      
      # The same to RV_[t-16d]
      AR22.model.t.16.coeffi <- as.numeric(AR22.model.coeffi.beta017.vector[p])
      
      # The same to RV_[t-17d]
      AR22.model.t.17.coeffi <- as.numeric(AR22.model.coeffi.beta018.vector[p])
      
      # The same to RV_[t-18d]
      AR22.model.t.18.coeffi <- as.numeric(AR22.model.coeffi.beta019.vector[p])
      
      # The same to RV_[t-19d]
      AR22.model.t.19.coeffi <- as.numeric(AR22.model.coeffi.beta020.vector[p])
      
      # The same to RV_[t-20d]
      AR22.model.t.20.coeffi <- as.numeric(AR22.model.coeffi.beta021.vector[p])
      
      # The same to RV_[t-21d]
      AR22.model.t.21.coeffi <- as.numeric(AR22.model.coeffi.beta022.vector[p])
      
      
      break
    }
    
    ## Repeat the above steps (STEP3-> STEP4-> STEP3) repeatedly 
    ## until the coefficients corresponding to the AR(22) model PART of HAR-infty-NN model 
    ## converge to a certain level. 
    
    
  }
  
  
  
  
########### STEP 5 : Fit the NN model lastly to the final.res data. 
  
  final.res <-(Rv.d.y2-AR22.model.t.coeffi*Rv.d.x1.2-AR22.model.t.1.coeffi*Rv.d.x1.2.1-AR22.model.t.2.coeffi*Rv.d.x1.2.2-AR22.model.t.3.coeffi*Rv.d.x1.2.3-AR22.model.t.4.coeffi*Rv.d.x1.2.4-AR22.model.t.5.coeffi*Rv.d.x1.2.5-AR22.model.t.6.coeffi*Rv.d.x1.2.6-AR22.model.t.7.coeffi*Rv.d.x1.2.7
               -AR22.model.t.8.coeffi*Rv.d.x1.2.8-AR22.model.t.9.coeffi*Rv.d.x1.2.9-AR22.model.t.10.coeffi*Rv.d.x1.2.10-AR22.model.t.11.coeffi*Rv.d.x1.2.11
               -AR22.model.t.12.coeffi*Rv.d.x1.2.12-AR22.model.t.13.coeffi*Rv.d.x1.2.13-AR22.model.t.14.coeffi*Rv.d.x1.2.14-AR22.model.t.15.coeffi*Rv.d.x1.2.15
               -AR22.model.t.16.coeffi*Rv.d.x1.2.16-AR22.model.t.17.coeffi*Rv.d.x1.2.17-AR22.model.t.18.coeffi*Rv.d.x1.2.18-AR22.model.t.19.coeffi*Rv.d.x1.2.19-AR22.model.t.20.coeffi*Rv.d.x1.2.20
               -AR22.model.t.21.coeffi*Rv.d.x1.2.21)
  
  ## 
  RVcombined.data3 <- cbind(final.res,Rv.d.x1.2,Rv.w.2,Rv.m.2)
  
  
  
  RVcombined.data3 <-as.data.frame(RVcombined.data3)
  
  
  
  NN.model.refitting.to.final.res <- neuralnet(final.res~Rv.d.x1.2+Rv.w.2+Rv.m.2,data=RVcombined.data3,
                                               hidden=10,linear.output=T,act.fct = "tanh")
  
  ################ tanh function values.
  
  
  a1 <- tanh(NN.model.refitting.to.final.res$weights[[1]][[1]][,1][1] + NN.model.refitting.to.final.res$weights[[1]][[1]][,1][2]*Rv.d.x1[nrow(RV.data)-i] + NN.model.refitting.to.final.res$weights[[1]][[1]][,1][3]*Rv.w[nrow(RV.data)-i]+
               NN.model.refitting.to.final.res$weights[[1]][[1]][,1][4]*Rv.m[nrow(RV.data)-i])
  
  a2 <- tanh(NN.model.refitting.to.final.res$weights[[1]][[1]][,2][1] + NN.model.refitting.to.final.res$weights[[1]][[1]][,2][2]*Rv.d.x1[nrow(RV.data)-i] + NN.model.refitting.to.final.res$weights[[1]][[1]][,2][3]*Rv.w[nrow(RV.data)-i]+
               NN.model.refitting.to.final.res$weights[[1]][[1]][,2][4]*Rv.m[nrow(RV.data)-i])
  
  a3 <- tanh(NN.model.refitting.to.final.res$weights[[1]][[1]][,3][1] + NN.model.refitting.to.final.res$weights[[1]][[1]][,3][2]*Rv.d.x1[nrow(RV.data)-i] + NN.model.refitting.to.final.res$weights[[1]][[1]][,3][3]*Rv.w[nrow(RV.data)-i]+
               NN.model.refitting.to.final.res$weights[[1]][[1]][,3][4]*Rv.m[nrow(RV.data)-i])
  
  a4 <- tanh(NN.model.refitting.to.final.res$weights[[1]][[1]][,4][1] + NN.model.refitting.to.final.res$weights[[1]][[1]][,4][2]*Rv.d.x1[nrow(RV.data)-i] + NN.model.refitting.to.final.res$weights[[1]][[1]][,4][3]*Rv.w[nrow(RV.data)-i]+
               NN.model.refitting.to.final.res$weights[[1]][[1]][,4][4]*Rv.m[nrow(RV.data)-i])
  
  a5 <- tanh(NN.model.refitting.to.final.res$weights[[1]][[1]][,5][1] + NN.model.refitting.to.final.res$weights[[1]][[1]][,5][2]*Rv.d.x1[nrow(RV.data)-i] + NN.model.refitting.to.final.res$weights[[1]][[1]][,5][3]*Rv.w[nrow(RV.data)-i]+
               NN.model.refitting.to.final.res$weights[[1]][[1]][,5][4]*Rv.m[nrow(RV.data)-i])
  
  a6 <- tanh(NN.model.refitting.to.final.res$weights[[1]][[1]][,6][1] + NN.model.refitting.to.final.res$weights[[1]][[1]][,6][2]*Rv.d.x1[nrow(RV.data)-i] + NN.model.refitting.to.final.res$weights[[1]][[1]][,6][3]*Rv.w[nrow(RV.data)-i]+
               NN.model.refitting.to.final.res$weights[[1]][[1]][,6][4]*Rv.m[nrow(RV.data)-i])
  
  a7 <- tanh(NN.model.refitting.to.final.res$weights[[1]][[1]][,7][1] + NN.model.refitting.to.final.res$weights[[1]][[1]][,7][2]*Rv.d.x1[nrow(RV.data)-i] + NN.model.refitting.to.final.res$weights[[1]][[1]][,7][3]*Rv.w[nrow(RV.data)-i]+
               NN.model.refitting.to.final.res$weights[[1]][[1]][,7][4]*Rv.m[nrow(RV.data)-i])
  
  a8 <- tanh(NN.model.refitting.to.final.res$weights[[1]][[1]][,8][1] + NN.model.refitting.to.final.res$weights[[1]][[1]][,8][2]*Rv.d.x1[nrow(RV.data)-i] + NN.model.refitting.to.final.res$weights[[1]][[1]][,8][3]*Rv.w[nrow(RV.data)-i]+
               NN.model.refitting.to.final.res$weights[[1]][[1]][,8][4]*Rv.m[nrow(RV.data)-i])
  
  a9 <- tanh(NN.model.refitting.to.final.res$weights[[1]][[1]][,9][1] + NN.model.refitting.to.final.res$weights[[1]][[1]][,9][2]*Rv.d.x1[nrow(RV.data)-i] + NN.model.refitting.to.final.res$weights[[1]][[1]][,9][3]*Rv.w[nrow(RV.data)-i]+
               NN.model.refitting.to.final.res$weights[[1]][[1]][,9][4]*Rv.m[nrow(RV.data)-i])
  
  a10 <- tanh(NN.model.refitting.to.final.res$weights[[1]][[1]][,10][1] + NN.model.refitting.to.final.res$weights[[1]][[1]][,10][2]*Rv.d.x1[nrow(RV.data)-i] + NN.model.refitting.to.final.res$weights[[1]][[1]][,10][3]*Rv.w[nrow(RV.data)-i]+
                NN.model.refitting.to.final.res$weights[[1]][[1]][,10][4]*Rv.m[nrow(RV.data)-i])
  
  
  ### All of the coefficients in the HAR-infty-NN model.
  
  
  coefficient.list <- list("Beta_01" = AR22.model.t.coeffi, "Beta_02" = AR22.model.t.1.coeffi, "Beta_03" = AR22.model.t.2.coeffi,
                           "Beta_04" = AR22.model.t.3.coeffi, "Beta_05" = AR22.model.t.4.coeffi, "Beta_06" = AR22.model.t.5.coeffi,
                           "Beta_07" = AR22.model.t.6.coeffi, "Beta_08" = AR22.model.t.7.coeffi, "Beta_09" = AR22.model.t.8.coeffi,
                           "Beta_010" = AR22.model.t.9.coeffi, "Beta_011" = AR22.model.t.10.coeffi, "Beta_012" = AR22.model.t.11.coeffi,
                           "Beta_013" = AR22.model.t.12.coeffi, "Beta_014" = AR22.model.t.13.coeffi, "Beta_015" = AR22.model.t.14.coeffi,
                           "Beta_016" = AR22.model.t.15.coeffi, "Beta_017" = AR22.model.t.16.coeffi, "Beta_018" = AR22.model.t.17.coeffi,
                           "Beta_019" = AR22.model.t.18.coeffi, "Beta_020" = AR22.model.t.19.coeffi, "Beta_021" = AR22.model.t.20.coeffi,
                           "Beta_022" = AR22.model.t.21.coeffi, "Beta_00" = NN.model.refitting.to.final.res$weights[[1]][[2]][[1]], 
                           
                           
                           "beta_1" = NN.model.refitting.to.final.res$weights[[1]][[2]][[2]], 
                           "gamma_10" = NN.model.refitting.to.final.res$weights[[1]][[1]][,1][1], "gamma_1d" = NN.model.refitting.to.final.res$weights[[1]][[1]][,1][2], "gamma_1w" = NN.model.refitting.to.final.res$weights[[1]][[1]][,1][3], 
                           "gamma_1m" = NN.model.refitting.to.final.res$weights[[1]][[1]][,1][4],
                           
                           "beta_2" = NN.model.refitting.to.final.res$weights[[1]][[2]][[3]], 
                           "gamma_20" = NN.model.refitting.to.final.res$weights[[1]][[1]][,2][1], "gamma_2d" = NN.model.refitting.to.final.res$weights[[1]][[1]][,2][2], "gamma_2w" = NN.model.refitting.to.final.res$weights[[1]][[1]][,2][3], 
                           "gamma_2m" = NN.model.refitting.to.final.res$weights[[1]][[1]][,2][4], 
                           
                           "beta_3" = NN.model.refitting.to.final.res$weights[[1]][[2]][[4]], 
                           "gamma_30" = NN.model.refitting.to.final.res$weights[[1]][[1]][,3][1], "gamma_3d" = NN.model.refitting.to.final.res$weights[[1]][[1]][,3][2], "gamma_3w" = NN.model.refitting.to.final.res$weights[[1]][[1]][,3][3], 
                           "gamma_3m" = NN.model.refitting.to.final.res$weights[[1]][[1]][,3][4], 
                           
                           "beta_4" = NN.model.refitting.to.final.res$weights[[1]][[2]][[5]], 
                           "gamma_40" = NN.model.refitting.to.final.res$weights[[1]][[1]][,4][1], "gamma_4d" = NN.model.refitting.to.final.res$weights[[1]][[1]][,4][2], "gamma_4w" = NN.model.refitting.to.final.res$weights[[1]][[1]][,4][3], 
                           "gamma_4m" = NN.model.refitting.to.final.res$weights[[1]][[1]][,4][4], 
                           
                           "beta_5" = NN.model.refitting.to.final.res$weights[[1]][[2]][[6]], 
                           "gamma_50" = NN.model.refitting.to.final.res$weights[[1]][[1]][,5][1], "gamma_5d" = NN.model.refitting.to.final.res$weights[[1]][[1]][,5][2], "gamma_5w" = NN.model.refitting.to.final.res$weights[[1]][[1]][,5][3], 
                           "gamma_5m" = NN.model.refitting.to.final.res$weights[[1]][[1]][,5][4],
  
                           
                           "beta_6" = NN.model.refitting.to.final.res$weights[[1]][[2]][[7]], 
                           "gamma_60" = NN.model.refitting.to.final.res$weights[[1]][[1]][,6][1], "gamma_6d" = NN.model.refitting.to.final.res$weights[[1]][[1]][,6][2], "gamma_6w" = NN.model.refitting.to.final.res$weights[[1]][[1]][,6][3], 
                           "gamma_6m" = NN.model.refitting.to.final.res$weights[[1]][[1]][,6][4],
  
                           "beta_7" = NN.model.refitting.to.final.res$weights[[1]][[2]][[8]], 
                           "gamma_70" = NN.model.refitting.to.final.res$weights[[1]][[1]][,7][1], "gamma_7d" = NN.model.refitting.to.final.res$weights[[1]][[1]][,7][2], "gamma_7w" = NN.model.refitting.to.final.res$weights[[1]][[1]][,7][3], 
                           "gamma_7m" = NN.model.refitting.to.final.res$weights[[1]][[1]][,7][4], 
  
                           "beta_8" = NN.model.refitting.to.final.res$weights[[1]][[2]][[9]], 
                           "gamma_80" = NN.model.refitting.to.final.res$weights[[1]][[1]][,8][1], "gamma_8d" = NN.model.refitting.to.final.res$weights[[1]][[1]][,8][2], "gamma_8w" = NN.model.refitting.to.final.res$weights[[1]][[1]][,8][3], 
                           "gamma_8m" = NN.model.refitting.to.final.res$weights[[1]][[1]][,8][4], 
  
                           "beta_9" = NN.model.refitting.to.final.res$weights[[1]][[2]][[10]], 
                           "gamma_90" = NN.model.refitting.to.final.res$weights[[1]][[1]][,9][1], "gamma_9d" = NN.model.refitting.to.final.res$weights[[1]][[1]][,9][2], "gamma_9w" = NN.model.refitting.to.final.res$weights[[1]][[1]][,9][3], 
                           "gamma_9m" = NN.model.refitting.to.final.res$weights[[1]][[1]][,9][4], 
  
                           "beta_10" = NN.model.refitting.to.final.res$weights[[1]][[2]][[11]], 
                           "gamma_100" = NN.model.refitting.to.final.res$weights[[1]][[1]][,10][1], "gamma_10d" = NN.model.refitting.to.final.res$weights[[1]][[1]][,10][2], "gamma_10w" = NN.model.refitting.to.final.res$weights[[1]][[1]][,10][3], 
                           "gamma_10m" = NN.model.refitting.to.final.res$weights[[1]][[1]][,10][4]  )

  
  
  
  #### Predicted value for RV_[T+1d]^(d), where T = nrow(RV.data)-i.
  predicted.values.for.RV.Tplus1d.d  <- AR22.model.t.coeffi*Rv.d.x1[nrow(RV.data)-i] + AR22.model.t.1.coeffi*Rv.d.x1[nrow(RV.data)-(i+1)]+
    AR22.model.t.2.coeffi*Rv.d.x1[nrow(RV.data)-(i+2)]+AR22.model.t.3.coeffi*Rv.d.x1[nrow(RV.data)-(i+3)]+AR22.model.t.4.coeffi*Rv.d.x1[nrow(RV.data)-(i+4)]+
    AR22.model.t.5.coeffi*Rv.d.x1[nrow(RV.data)-(i+5)]+AR22.model.t.6.coeffi*Rv.d.x1[nrow(RV.data)-(i+6)]+
    AR22.model.t.7.coeffi*Rv.d.x1[nrow(RV.data)-(i+7)]+AR22.model.t.8.coeffi*Rv.d.x1[nrow(RV.data)-(i+8)]+AR22.model.t.9.coeffi*Rv.d.x1[nrow(RV.data)-(i+9)]
  +AR22.model.t.10.coeffi*Rv.d.x1[nrow(RV.data)-(i+10)]+AR22.model.t.11.coeffi*Rv.d.x1[nrow(RV.data)-(i+11)]+AR22.model.t.12.coeffi*Rv.d.x1[nrow(RV.data)-(i+12)]
  +AR22.model.t.13.coeffi*Rv.d.x1[nrow(RV.data)-(i+13)]+AR22.model.t.14.coeffi*Rv.d.x1[nrow(RV.data)-(i+14)]+AR22.model.t.15.coeffi*Rv.d.x1[nrow(RV.data)-(i+15)]
  +AR22.model.t.16.coeffi*Rv.d.x1[nrow(RV.data)-(i+16)]+AR22.model.t.17.coeffi*Rv.d.x1[nrow(RV.data)-(i+17)]+AR22.model.t.18.coeffi*Rv.d.x1[nrow(RV.data)-(i+18)]
  +AR22.model.t.19.coeffi*Rv.d.x1[nrow(RV.data)-(i+19)]+AR22.model.t.20.coeffi*Rv.d.x1[nrow(RV.data)-(i+20)]
  +AR22.model.t.21.coeffi*Rv.d.x1[nrow(RV.data)-(i+21)]+NN.model.refitting.to.final.res$weights[[1]][[2]][[1]]+NN.model.refitting.to.final.res$weights[[1]][[2]][[2]]*a1+NN.model.refitting.to.final.res$weights[[1]][[2]][[3]]*a2+
    NN.model.refitting.to.final.res$weights[[1]][[2]][[4]]*a3+NN.model.refitting.to.final.res$weights[[1]][[2]][[5]]*a4+NN.model.refitting.to.final.res$weights[[1]][[2]][[6]]*a5
  +NN.model.refitting.to.final.res$weights[[1]][[2]][[7]]*a6+NN.model.refitting.to.final.res$weights[[1]][[2]][[8]]*a7+NN.model.refitting.to.final.res$weights[[1]][[2]][[9]]*a8
  +NN.model.refitting.to.final.res$weights[[1]][[2]][[10]]*a9+NN.model.refitting.to.final.res$weights[[1]][[2]][[11]]*a10
  
  
  if(printcoeffi=="YES"){
    print(coefficient.list)
  }
  
  else if(printcoeffi=="NO"){
    print(predicted.values.for.RV.Tplus1d.d)
  }
  
  else if(printcoeffi=="BOTH"){
    print(coefficient.list)
    print(predicted.values.for.RV.Tplus1d.d)
  }
  
  
  else
    print("You might have a typo error in 'printcoeffi' argument")
  
}


# HAR_AR22_NN.tanh(KOSPI.0615.data.RV,100,200,"YES")




##### HAR-AR(22)-NN forecasting function.

HAR_AR22_NN_forecast <- function(RV.data,i,r,acti.fun){
  
  if(acti.fun=="logistic"){
    
    ########### Predicted value for RV_[T+1d]^(d), where T = nrow(RV.data)-i
    
    HAR_AR22_NN.sig(KOSPI.0615.data.RV,i,r,"NO")
    
  }
  else if(acti.fun=="tanh"){
    
    ########### Predicted value for RV_[T+1d]^(d), where T = nrow(RV.data)-i 
    
    HAR_AR22_NN.tanh(KOSPI.0615.data.RV,i,r,"NO")
  }
  
  else
    print("You might have a typo error in 'acti.fun' argument")
}


#HAR_AR22_NN_forecast(KOSPI.0615.data.RV,100,200,"logistic")

#HAR_AR22_NN_forecast(KOSPI.0615.data.RV,100,200,"tanh")

#HAR_infty_NN_forecast(KOSPI.0615.data.RV,100,200,"asdf")







  