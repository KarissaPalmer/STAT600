#include <RcppArmadillo.h>

using namespace Rcpp;
// [[Rcpp::depends(RcppArmadillo)]]

namespace helper{
double lp(arma::vec x, double theta){
  arma::vec num = 2*(x-theta);
  arma::vec denom = 1+arma::pow((x-theta),2);
  double lprime = arma::sum(num/denom);
  return lprime; }

double lpp(arma::vec x, double theta){
  arma::vec num = -2+2*arma::pow(x-theta,2);
  arma::vec denom = arma::pow(1+arma::pow(x-theta,2),2);
  double lprime2 = arma::sum(num/denom);
  return lprime2;
}

arma::vec lp_mat(arma::mat X, arma::vec y, arma::vec beta){
  int n = X.n_rows;
  //vector of ones 
  arma::vec z(n, arma::fill::ones);
  //join with another vector
  X = arma::join_rows(z,X);
  
  arma::mat p = (exp(X * beta)/(1+exp(X * beta)));
  
  arma::vec grad = X.t() * (y - p);
  return grad;
}

arma::mat hess(arma::mat X, arma::vec beta){
  int n = X.n_rows;
  //vector of ones 
  arma::vec z(n, arma::fill::ones);
  //join with another vector
  X = arma::join_rows(z,X);
  //Initialize p
  arma::mat p = (exp(X * beta)/(1+exp(X * beta)));
  
  //Make p into matrix
  arma::mat W = diagmat(p * (1-p).t() );
  
  arma::mat grad2 = X.t() * W * X;
  return grad2;
}

}

// [[Rcpp::export]]
Rcpp::List bisect(arma::vec x, double a, double b, double tol){ 
  if (a >= b){
    stop("'b' must be greater than a");
  }
  if (helper::lp(x,a) * helper::lp(x,b) >= 0){
    // stop("There is no root in the supplied interval.");
  }
  unsigned int iter = 0;
  double x0 = 0;
  double lp0 = 0;
  
  iter += 1;
  while(fabs(b-a) > tol){
    x0 = (a+b)/2;
    lp0 = helper::lp(x, x0);
    
    if(lp0 == 0 | iter > 2000){
      break;
    }else if(helper::lp(x,x0)*helper::lp(x,a) < 0){
      b = x0;
      iter += 1;
    }else{
      a = x0;
      iter += 1;
    }
    // iter = iter + 1;
  }
  
  Rcpp::List output;
  output["x0"] = x0;
  output["iter"] = iter;
  return output;
}



// Newton-Raphson
// [[Rcpp::export]]
Rcpp::List newtr(arma::vec x, double theta0, double tol){
  unsigned int iter = 0;
  double lp0 = helper::lp(x,theta0);
  double lpp0 = helper::lpp(x,theta0);
  double rat = lp0/lpp0;
  
  iter += 1;
  double theta1 = theta0 - rat;
  
  //Just use theta0 in place of some difference
  while(fabs(theta1-theta0)> tol){
    if(lp0 == 0 | iter > 2000){
      break;
    }else if(lpp0 == 0){
      stop("The second derivative is zero at one of the iterated theta0s.");
    }else{
      theta0 = theta1;
      lp0 = helper::lp(x,theta0);
      lpp0 = helper::lpp(x,theta0);
      rat = lp0/lpp0;
      iter += 1;
      theta1 = theta0 - rat;
      // std::cout << "theta1:" << theta1 << std::endl;
    }
  }
  
  
  Rcpp::List output;
  output["theta1"] = theta1;
  output["iter"] = iter;
  return output;
}



// Fisher Scoring
// [[Rcpp::export]]
Rcpp::List fish_score(arma::vec x, double theta0, double tol){
  double n = x.n_rows;
  unsigned int iter = 0;
  double lp0 = helper::lp(x,theta0);
  double rat = (2/n)*lp0;
  
  iter += 1;
  double theta1 = theta0 + rat;
  
  //Just use theta0 in place of some difference
  while(abs(theta1-theta0) > tol){
    if(lp0 == 0 | iter > 2000){
      break;
    }
    else{
      theta0 = theta1;
      lp0 = helper::lp(x,theta0);
      rat = (2/n)*lp0;
      iter += 1;
      theta1 = theta0 + rat;
    }
  }
  
  
  Rcpp::List output;
  output["theta1"] = theta1;
  output["iter"] = iter;
  return output;
}


// Secant Method
// [[Rcpp::export]]
Rcpp::List secant(arma:: vec x, double theta0, double theta1, double tol){
  unsigned int iter = 0;
  double lp0 = helper::lp(x,theta0);
  double lp1 = helper::lp(x,theta1);
  
  double rat = (theta1-theta0)/(lp1-lp0);
  iter += 1;
  double theta2 = theta1 - lp1 * rat;
  
  while(abs(theta2-theta1) > tol){
    if(lp0 == 0 | iter > 200 | lp1 == 0){
      stop("Does not converge in under 200 iterations.");
    }else{
      iter += 1;
      theta0 = theta1;
      theta1 = theta2;
      
      lp0 = helper::lp(x,theta0);
      lp1 = helper::lp(x,theta1);
      rat = (theta1-theta0)/(lp1-lp0);
      theta2 = theta1 - lp1 * rat;
    }
  }
  Rcpp::List output;
  output["theta"] = theta2;
  output["iter"] = iter;
  return output;
}


//Newton-Raphson for matrix
// [[Rcpp::export]]
Rcpp::List newtrmat(arma::mat X, arma::vec y, arma::vec beta0, double tol){
  //Start iterations
  unsigned int iter = 0;
  //Find the inverse of the hessian at initial vector
  arma::mat hess1 = inv(helper::hess(X,beta0));
  //Gradient at initial vector
  arma::mat grad1 = helper::lp_mat(X, y, beta0);
  //First iteration of finding beta
  arma::mat beta = beta0 + hess1*grad1;
  //Find distance for iterations checks
  double dist = sqrt(sum(arma::pow(beta-beta0,2)));
  
  //set initialize iterations
  iter += 1;
  
  // Initialize while loop
  while(dist > tol){
    if (iter > 2000){
      //Set max # of iterations
      stop("Number of iterations is too high.");
    }else{
      //Go through each of the steps
      beta0 = beta;
      iter += 1;
      hess1 = inv(helper::hess(X,beta0));
      grad1 = helper::lp_mat(X, y, beta0);
      beta = beta0 + hess1*grad1;
      
      //Update distances between points
      dist = sqrt(sum(arma::pow(beta-beta0,2)));
    }
  }
  
  Rcpp::List output;
  output["iter"] = iter;
  output["Beta"] = beta;
  return output;
}
