true_vs_fitted_plot <- function(true_vals, fitted_vals) {
  df <- cbind(as.numeric(true_vals), as.numeric(fitted_vals)) %>% as.data.frame()
  colnames(df) <- c("True_vals", "Fitted")
  
  ggplot(data=df, mapping = aes(x=True_vals, y=Fitted)) +
    geom_point() +
    geom_abline(intercept = 0, slope = 1) +
    xlab("Data (actual values)") +
    ylab("Fitted values") +
    theme_light()
}


residuals_against_predictors_plot <- function(df, predictor_lbls, residuals_lbl) {
  
  plots <- lapply(predictor_lbls, function(lbl) {
    ggplot(mapping = aes(x=df[[lbl]], y=df[[residuals_lbl]])) +
      geom_point() +
      xlab(lbl) + ylab(residuals_lbl) +
      theme_light()
  })
  
  require(ggpubr)
  ggarrange(plotlist = plots, ncol = 2, nrow = round(length(plots)/2))
  
}