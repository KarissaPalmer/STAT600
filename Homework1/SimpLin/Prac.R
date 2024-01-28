library(Rcpp)
library(devtools)

uninstall()
compileAttributes()
build()
install(build_vignettes = TRUE)
library(SimpLin)


