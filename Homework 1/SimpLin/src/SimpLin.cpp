#include <RcppArmadillo.h>
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]
List SimpLinCpp(arma::vec & x, arma::vec & y){
  int n = x.n_rows;
  arma::mat X(n,2);
  for( int i = 0; i < n; i++){
    X(i,0) = 1;
    X(i,1) = x(i,0);
  }
  
  arma::mat invDes = inv(X.t() * X);
  arma::mat beta = invDes * X.t() * y;
  
  // Predicted values
  arma::vec pred = X * beta;
  
  // Residuals
  arma::vec resid = y - pred;
  
  // Standard error
  double mse = arma::accu(arma::pow(resid,2))/(n - 2);
  arma::mat semat = arma::sqrt(mse * invDes);
  arma::vec coefSE = semat.diag();
  
  //95% CI
  arma::mat conf_int(2,2);
  double tstat = R::qt(0.975, n-2,1,0);
  conf_int(0,0) = beta(0,0)-tstat*coefSE(0,0);
  conf_int(0,1) = beta(0,0)+tstat*coefSE(0,0);
  conf_int(1,0) = beta(1,0)-tstat*coefSE(1,0);
  conf_int(1,1) = beta(1,0)+tstat*coefSE(1,0);
  
  
  return List::create(Named("Coefficients") = beta,
                      Named("SEs") = coefSE,
                      Named("Conf_Ints") = conf_int,
                      Named("Residuals") = resid,
                      Named("Pred_Vals") = pred);
}

