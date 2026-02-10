# Helper functions: Experiment 1
# Participants choose gambles sequentially

# Function used to determine which participants to remove based on 3 factors:
# 1. Participants did not complete the task exactly one time
# 2. Participants selected the decoy option at or more than chance level when first compared to target It is irrational to select the decoy at all, 
# so selecting the decoy more at or than chance indicates participants did not pay attention/ take task seriously.
# 3. Participants only selected 1 key, 2 key, 3 key, or no key, showing lack of engagement in task. 
# Note: no one was removed for option 3, so did not include analysis in paper. 
determine_participants_to_remove <- function(data){
  # Did not complete task/ completed task more than once
  # Did not complete all trials
  choices_completed <- data %>% filter(practice_trial == 0) %>% group_by(id) %>% tally()
  num_trials <- median(choices_completed$n)
  dropouts1 <- choices_completed %>% filter(n != num_trials) 
  
  # Failed threshold
  threshold <- -1.64*0.33/(sqrt(num_trials *1/3)) + 0.33 # Only 1/3 of trials is between target vs decoy
  dropouts2 <- data %>% # Only consider trials where target and decoy are first comparison
    filter(practice_trial == 0 & 
             ((trial_set == "riskyDecoy" & added_gamble == "S") | 
                (trial_set == "safeDecoy" & added_gamble == "R"))) %>%
    group_by(id) %>%
    summarise(decoy_chosen = mean(first_response_gamble == "D")) %>%
    filter(decoy_chosen >= threshold) %>%
    select(id)
  
  # Remove participants who had pattern of selecting options
  dropouts3 <- data %>%
    group_by(id) %>%
    summarise(avg_ff = mean(first_response == "f" & final_response == "f"),
              avg_jj = mean(first_response == "j" & final_response == "j"),
              avg_fj = mean(first_response == "f" & final_response == "j"),
              avg_jf = mean(first_response == "j" & final_response == "j"),
              avg_no_response = mean(final_response == "no_response")) %>%
    filter(avg_ff == 1 | avg_ff == 0 | avg_fj == 1 | avg_fj == 0 | avg_jf == 1 | avg_jf == 0 | avg_jj == 1 | avg_jj == 0 | avg_no_response == 1) %>%
    select(id)
  
  # Not pre-registered
  dropouts4 <- data %>%
    group_by(id) %>%
    summarise(avg_same = mean(first_response == final_response)) %>%
    filter(avg_same > 0.8) %>%
    select(id)
  
  print("Breakdown of dropouts")
  print(paste("Did not complete task more than once:", length(dropouts1$id), sep = " "))
  print(paste("Selected decoy option more than chance:", length(dropouts2$id), sep = " "))
  print(paste("Failed to engage in task:", length(dropouts3$id), sep = " "))
  print(paste("Selected same key on trial more than 80%:", length(dropouts4$id), sep = " "))
  print("-----------------------------------------------")
  dropouts <- c(dropouts1$id, dropouts2$id, dropouts3$id, dropouts4$id)
  print(paste("Total number of dropouts:", length(dropouts), sep = " "))
  return(dropouts)
}

clean_data_frame <- function(data, dropouts){
  print("Percentage of no choices")
  no_choice <- data %>%
    summarise(no_choice = mean(made_final_choice == 0))
  print(no_choice)

  df <- data %>%
    filter(!(id %in% dropouts) & made_first_choice == 1 & made_final_choice == 1 & practice_trial == 0) %>%
    # Change decoy order
    mutate(decoy_order = case_when(
      trial_set == "riskyDecoy" ~ chartr("RS", "TC", order), # R->T, S->C
      trial_set == "safeDecoy"  ~ chartr("RS", "CT", order), # R->C, S->T
      TRUE ~ order
    ),
    # Create decoy first response 
    decoy_first_response_gamble = case_when(
      trial_set == "riskyDecoy" & first_response_gamble == "R" ~ "T",
      trial_set == "riskyDecoy" & first_response_gamble == "S" ~ "C",
      trial_set == "safeDecoy"  & first_response_gamble == "R" ~ "C",
      trial_set == "safeDecoy"  & first_response_gamble == "S" ~ "T",
      TRUE ~ first_response_gamble   # D does not need to change
    ),
    # Create decoy first left gamble 
    decoy_first_left_gamble = case_when(
      trial_set == "riskyDecoy" & first_left_gamble == "R" ~ "T",
      trial_set == "riskyDecoy" & first_left_gamble == "S" ~ "C",
      trial_set == "safeDecoy"  & first_left_gamble == "R" ~ "C",
      trial_set == "safeDecoy"  & first_left_gamble == "S" ~ "T",
      TRUE ~ first_left_gamble  
    ),
    # Create decoy first right gamble 
    decoy_first_right_gamble = case_when(
      trial_set == "riskyDecoy" & first_right_gamble == "R" ~ "T",
      trial_set == "riskyDecoy" & first_right_gamble == "S" ~ "C",
      trial_set == "safeDecoy"  & first_right_gamble == "R" ~ "C",
      trial_set == "safeDecoy"  & first_right_gamble == "S" ~ "T",
      TRUE ~ first_right_gamble  
    ),
    # Create decoy final response 
    decoy_final_response_gamble = case_when(
      trial_set == "riskyDecoy" & final_response_gamble == "R" ~ "T",
      trial_set == "riskyDecoy" & final_response_gamble == "S" ~ "C",
      trial_set == "safeDecoy"  & final_response_gamble == "R" ~ "C",
      trial_set == "safeDecoy"  & final_response_gamble == "S" ~ "T",
      TRUE ~ final_response_gamble   
    ),
    # Create decoy final left gamble 
    decoy_final_left_gamble = case_when(
      trial_set == "riskyDecoy" & final_left_gamble == "R" ~ "T",
      trial_set == "riskyDecoy" & final_left_gamble == "S" ~ "C",
      trial_set == "safeDecoy"  & final_left_gamble == "R" ~ "C",
      trial_set == "safeDecoy"  & final_left_gamble == "S" ~ "T",
      TRUE ~ final_left_gamble  
    ),
    # Create decoy final right gamble 
    decoy_final_right_gamble = case_when(
      trial_set == "riskyDecoy" & final_right_gamble == "R" ~ "T",
      trial_set == "riskyDecoy" & final_right_gamble == "S" ~ "C",
      trial_set == "safeDecoy"  & final_right_gamble == "R" ~ "C",
      trial_set == "safeDecoy"  & final_right_gamble == "S" ~ "T",
      TRUE ~ final_right_gamble  
    ),
    # Added gamble
    decoy_added_gamble = case_when(
      trial_set == "riskyDecoy" & added_gamble == "R" ~ "T",
      trial_set == "riskyDecoy" & added_gamble == "S" ~ "C",
      trial_set == "safeDecoy"  & added_gamble == "R" ~ "C",
      trial_set == "safeDecoy"  & added_gamble == "S" ~ "T",
      TRUE ~ added_gamble  
    ))
  return(df)
}


# Note that this function is only used for annotating the graphs
convert_annotations <- function(t_results) {
  # Annotations for plot
  annotations <- t_results %>%
    mutate(annotation = case_when(
      p.adj > 0.05 ~ "n.s.",
      p.adj <= 0.05 & p.adj > 0.01 ~ "*",
      p.adj <= 0.01 & p.adj >= 0.001 ~ "**",
      p.adj < 0.001 ~ "***",
    ))
  return(annotations)
}

sequential_decoy_effect <- function(choices_data_set){
  num_participants <- length(unique(choices_data_set$id))
  # Calculate chose target for each participant
  general_decoy_effect <- choices_data_set %>%
    filter(final_response_gamble != "D") %>%
    group_by(id, trial_set)%>%
    summarise(chose_risky = mean(final_response_gamble == "R")) %>%
    ungroup()
  
  decoy_selection_rate <- choices_data_set %>%
        summarise(prop_decoy = mean(final_response_gamble == "D"))
  print("decoy selection rate:")
  print(decoy_selection_rate)
  # Run one-sided t-test  
  t_test_result <- general_decoy_effect %>% 
    t_test(
      chose_risky ~ trial_set,
      paired = TRUE,
      alternative = "greater"  # riskyDecoy is group 1
    ) %>%
    mutate(p.adj = p) # Does not mean anything but added column so annotations code works. 
  print("t-test results")
  print(t_test_result)
  
  
  # Plot results
  # This is to calculate standard deviation
  differences <- general_decoy_effect %>%  group_by(id) %>% summarise(avg_by_ind = mean(chose_risky))
  
  # Calculate chose target for each trial type
  plot_general_decoy_effect <- general_decoy_effect %>%
    left_join(differences, by = "id") %>%
    mutate(avg_no_variance = chose_risky - avg_by_ind) %>%
    group_by(trial_set)%>%
    summarise(chose_risky_prop = mean(chose_risky), 
              standard_deviation = sd(chose_risky),
              standard_error = sd(avg_no_variance)/sqrt(num_participants)) 
  print("mean and sd or risky selections")
  print(plot_general_decoy_effect)
  
  annotations <- convert_annotations(t_test_result)
  ymax_label = max(plot_general_decoy_effect$chose_risky_prop)

  
  
  # Plot decoy effect results
  plot_general_decoy_effect$trial_numeric <- c(0.8, 1)
  p <- ggplot(plot_general_decoy_effect, aes(x = trial_numeric, y = chose_risky_prop, group = trial_numeric)) +
    geom_bar(stat = "identity", fill = "#928b8b", col = "black", width = 0.15) +
    geom_errorbar(aes(ymin = chose_risky_prop - standard_error,
                      ymax = chose_risky_prop + standard_error),
                  width = 0.04, size = 1) +
    labs(x = "Trial type", y = "P(Risky)") +
    scale_x_continuous(breaks = c(0.8, 1),
                       labels = c("Risky-decoy", "Safe-decoy"),
                       limits = c(0.7, 1.5),   # tightens white space and centers bars
                       expand = c(0, 0)) +
    scale_y_continuous(breaks = seq(0, 0.6, 0.1),
                       limits = c(0, 0.7),
                       expand = c(0, 0)) +
    theme(panel.grid = element_blank(),
          panel.background = element_blank(),
          axis.line = element_line(colour = "black", size = 1),
          axis.ticks = element_line(colour = "black", size = 1),
          axis.ticks.length = unit(0.3, "cm"),
          axis.text = element_text(size = 22, color = "black"),
          axis.title = element_text(size = 26, color = "black"),
          axis.title.x = element_text(size = 26, color = "black", hjust =0.225),
          legend.position = "none")+
    geom_signif(xmin = 0.8, xmax = 1,
                annotation=annotations$annotation[1], 
                y_position = ymax_label + 0.05, textsize = 6, tip_length = 0.2)
  return(p) 
}

# Ordering effects
ordering_effect <- function(choices_data_set){
  num_participants <- length(unique(choices_data_set$id))
  # Selecting target based on added gamble
  ordering_data <- choices_data_set %>%
    group_by(id, decoy_added_gamble) %>%
    filter(final_response_gamble != "D") %>%
    summarise(chose_target = mean(decoy_final_response_gamble == "T")) %>% 
    ungroup()
  
  anova_results <- get_anova_table(ordering_data %>% anova_test(dv = chose_target, within = decoy_added_gamble, wid = id))
  print(anova_results)
  pairwise_results <- ordering_data %>%
    group_by(decoy_added_gamble) %>%
    t_test(chose_target ~ 1, mu = 0.5, alternative = "two.sided") %>%
    adjust_pvalue(method = "bonferroni") %>%
    add_significance()
  print(pairwise_results)
  
  
  # Calculate chose target for each trial type
  plot_ordering_data  <- ordering_data %>%
    group_by(decoy_added_gamble)%>%
    summarise(chose_target_prop = mean(chose_target), 
              standard_error = sd(chose_target)/sqrt(n())) 
  
  annotations <- convert_annotations(pairwise_results)
  ymax_label = max(plot_ordering_data$chose_target_prop) + 0.05
  
  plot_ordering_data$decoy_added_gamble_numeric <- c(0.8, 1, 1.2)
  
  p <- ggplot(plot_ordering_data, aes(x = decoy_added_gamble_numeric, y = chose_target_prop, group = decoy_added_gamble_numeric)) +
    geom_bar(stat = "identity", fill = "#928b8b", col = "black", width = 0.15) +
    geom_errorbar(aes(ymin = chose_target_prop - standard_error, ymax = chose_target_prop + standard_error), 
                  size = 1, width = 0.04) +
    scale_x_continuous(breaks = c(0.8, 1, 1.2),
                       labels = c("Target-decoy", "Competitor-target", "Competitor-decoy"),
                       limits = c(0.7, 1.5),
                       expand = c(0, 0)) +
    labs(x = "First comparison", y = "P(Target)") +
    theme(panel.grid = element_blank(),
          panel.background = element_blank(),
          axis.line = element_line(colour = "black", size = 1),
          axis.ticks = element_line(colour = "black", size = 1),
          axis.ticks.length = unit(0.3, "cm"),
          axis.text = element_text(size = 22, color = "black"),
          axis.title = element_text(size = 26, color = "black", face = "plain"),
          axis.title.x = element_text(size = 26, color = "black", hjust = 0.35),
          legend.position = "none") +
    scale_y_continuous(breaks = seq(0, 0.69, by = 0.1), expand = c(0, 0), limits = c(0, 0.7)) +
    geom_hline(yintercept = 0.5, color = "black", linetype = "dashed") +
    annotate("text", x = c(0.8, 1, 1.2), y = c(ymax_label, ymax_label, ymax_label), 
             label = annotations$annotation, size = 12, vjust = 1)
  return(p)
}

first_option_correlation <- function(choices_data_set, delta_df){
  ordering_data <- choices_data_set %>%
    filter(decoy_final_response_gamble != "D") %>%
    group_by(id, decoy_added_gamble) %>%
    summarise(chose_target = mean(decoy_final_response_gamble == "T")) %>% 
    ungroup()
  
  wide_data <- ordering_data %>%
    pivot_wider(
      names_from = decoy_added_gamble,
      values_from = chose_target
    ) %>%
    left_join(delta_df, by = "id")
  
  cor_test <- cor.test(wide_data$C, wide_data$T)
  print(cor_test)
  for_poster = 0
  for_poster2 = 0
  
  
  p <- ggplot(wide_data, aes(x = C, y = T, color = delta)) +
    geom_point(alpha = 1, size = 3) +
    scale_y_continuous(limits = c(0.25, 0.85), breaks = seq(0.1, 1.0, by = 0.1))+
    geom_hline(yintercept = 0.5, alpha = 0.7, linetype = "dashed", color = "#b1a59c", size = 0.5) +
    geom_vline(xintercept = 0.5, linetype = "dashed", color = "#b1a59c", size = 0.5) +
    geom_smooth(method = "lm", se = TRUE, color = "#1a1a1a", fill = "#b1a59c") +
    annotate("text", x = 0.25, y = 0.85, label = "Target favored only\n when CD first", size = 5+for_poster2, hjust = 0.5) +
    annotate("text", x = 0.60, y = 0.27, label = "Target favored only\n when TD first", size = 5+for_poster2, hjust = 0.5) +
    annotate("text", x = 0.60, y = 0.85, label = "Target favored \n when CD and TD first", size = 5+for_poster2, hjust = 0.5) +
    annotate("text", x = 0.25, y = 0.27, label = "Target never favored", size = 5+for_poster2, hjust = 0.5) +
    scale_color_gradient2(low = "#005b88", mid = "#b19cd9", high = "#9f2632") +
    labs(
      # title = "Correlation between Chose Target based on whether Competitor Decoy vs Target Decoy shown first",
      x = "P(Target) - Target and decoy first",
      y = "P(Target) - Competitor and decoy first",
      color = "Delta") +
    theme_minimal() +
    theme(panel.grid.major = element_blank(), 
          panel.grid.minor = element_blank(),
          axis.line = element_line(color = "black", size = 1),
          axis.ticks = element_line(color = "black", size = 1),
          axis.ticks.length = unit(0.3, "cm"),
          axis.text = element_text(color = "black", size = 18+ for_poster),
          axis.title = element_text(color = "black", size = 20+ for_poster),
          legend.title = element_text(size = 20+ for_poster),
          legend.text = element_text(size = 18+ for_poster),
          legend.key.size = unit(1, "cm"))
  return(p)
}

ordering_effects_by_delta <- function(choices_data_set, delta_df){
  delta_percentile <- delta_df %>%
    mutate(delta_tercile = ntile(delta, 3),  
           delta_group = factor(delta_tercile,
                                levels = c(1, 2, 3),
                                labels = c("Low D", "Medium D", "High D")))
  
  ordering_data <- choices_data_set %>%
    group_by(id, decoy_added_gamble) %>%
    filter(final_response_gamble != "D") %>%
    summarise(chose_target = mean(decoy_final_response_gamble == "T"), .groups = "drop") %>%
    left_join(delta_percentile, by = "id")
  
  pairwise_results <- ordering_data %>%
    group_by(decoy_added_gamble, delta_group) %>%
    t_test(chose_target ~ 1, mu = 0.5, alternative = "two.sided") %>%
    adjust_pvalue(method = "bonferroni") %>%
    add_significance()
  
  print(pairwise_results)
  annotations <- convert_annotations(pairwise_results)
  
  plot_ordering_data  <- ordering_data %>%
    group_by(decoy_added_gamble, delta_group, .drop = FALSE) %>%
    summarise(chose_target_prop = mean(chose_target), 
              standard_error = sd(chose_target)/sqrt(n()), .groups = "drop")  # Changed here
  
  plot_with_annotations <- plot_ordering_data %>%
    left_join(annotations, by = c("decoy_added_gamble", "delta_group")) %>%
    mutate(label_y = chose_target_prop + standard_error + 0.02)
  
  p <- ggplot(plot_ordering_data, aes(x = decoy_added_gamble, y = chose_target_prop)) +
    geom_bar(stat = "identity", position = position_dodge(), fill = "#928b8b", col = "black") +
    geom_errorbar(aes(ymin = chose_target_prop - standard_error, 
                      ymax = chose_target_prop + standard_error), 
                  size = 1, width = 0.3, position = position_dodge(.9)) +
    scale_x_discrete(labels = c("(T,D)", "(C,T)", "(C,D)")) +
    labs(x = "First comparison", y = "P(Target)") +
    theme(panel.grid.major = element_blank(), 
          panel.grid.minor = element_blank(), 
          strip.background = element_blank(), 
          strip.text = element_text(size = 24), 
          panel.background = element_blank(), 
          axis.line = element_line(colour = "black"), 
          axis.text = element_text(size = 22, colour = "black"), 
          axis.title = element_text(size = 28)) +
    scale_y_continuous(breaks = seq(0, 0.79, by = 0.1), expand = c(0, 0), limits = c(0, .71)) +
    geom_hline(yintercept = 0.5, color = "#928b8b", linetype = "dashed") +
    facet_wrap(~ delta_group, nrow = 1) +
    geom_text(data = plot_with_annotations, 
              aes(x = decoy_added_gamble, y = label_y, label = annotation),
              size = 8, vjust = 0)
  return(p)
}

rt_delta_analysis <- function(choices_data_set, delta_df){
  rt_model <- choices_data_set %>%
    filter(decoy_final_response_gamble != "D") %>%
    left_join(delta_df, by = "id") %>%
    select(id, delta, first_reaction_time, final_reaction_time, decoy_final_response_gamble, decoy_first_response_gamble) %>%
    mutate(repeated_choice = ifelse(decoy_final_response_gamble == decoy_first_response_gamble, 1, -1),
           centered_delta = delta - mean(delta))
  
  m <- lmer(final_reaction_time ~ centered_delta * repeated_choice + (1|id), data = rt_model)
  print(summary(m))
  
  rt_model <- rt_model %>%
    mutate(choice_type = ifelse(repeated_choice == 1, "Maintained", "Switched"))
  
  p <- ggplot(rt_model, aes(x = centered_delta, y = final_reaction_time, color = choice_type)) +
   # geom_point(alpha = 0.3) +
    geom_smooth(method = "lm", se = TRUE, size = 1.5, show.legend = TRUE) +
    scale_color_manual(
      values = c("Maintained" = "#98cea6", "Switched" = "#e8943d"),
      labels = c("Maintained" = "Repeated choice", "Switched" = "Switched choice")
    ) +
    labs(
      x = "Centered D",
      y = "Final Reaction Time (ms)"
    ) +
    guides(color = guide_legend(ncol = 1)) +  # <-- side-by-side legend
    theme_minimal() +
    theme(
      legend.position = c(0.2, .95),
      legend.direction = "vertical",
      legend.title = element_blank(),
      legend.background = element_blank(),
      legend.key = element_blank(),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      legend.text = element_text(size = 18),
      axis.text = element_text(size = 18),
      axis.title = element_text(size = 20),
      axis.ticks = element_line(color = "black", linewidth = 0.6),
      axis.ticks.length = unit(5, "pt"),
      axis.line = element_line(color = "black")
    )
  

  return(p)
}


