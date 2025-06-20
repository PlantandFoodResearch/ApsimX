# library
library(tidyverse)
#--------------------------------------------------------------------------------
# WheatSSER.apsimx in APSIM UI needs to be used to create WheatSSER.Output.csv 
# by exporting DataStore to text files
# -------------------------------------------------------------------------------

# Read CSV file
data <- read.csv(
  file = "C:/github/ApsimX/Prototypes/Wheat_NZ_SSER/WheatSSER.Output.csv",
  header = TRUE,
  stringsAsFactors = TRUE
)

# Display structure of the data
str(data)


# Display structure of the data
head(data)

# clean data
work_data <- data %>%
  filter(Experiment == "SimulationExp") %>%
  separate(Location, into = c("Location", "Climate"), sep = "_", remove = TRUE)
  
summary(work_data)

# plot data
work_data %>%
  filter(!(Septoria == "Clean" & Genotype == "Susceptible")) %>%
  ggplot(aes(x=interaction(Genotype,Septoria) , 
             y=GrainYield_kgPerHa, colour=interaction(Genotype,Septoria))) +
  geom_violin(alpha=0.5) +
  geom_boxplot(alpha=0.5) +
  geom_jitter(width = 0, alpha = 0.5) +
  facet_grid(Location~Climate, scale="free")+
  #theme_bw() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1)
  ) +
  ylab("Simulated wheat grain yield\n (kg/ha per year)") +
  xlab("Simulation Treatment")+
  theme(legend.position = "none", legend.title = element_blank()) +
  scale_x_discrete(labels = c("Resistant.Clean" = "No infection", 
                              "Resistant.Infected" = "STB Resistant",
                              "Susceptible.Infected" = "STB Susceptible"))

# summary
work_data_summary <- work_data %>%
  group_by(Year, Septoria, Location, Climate) %>%
    summarise(MedianYield = median(GrainYield_kgPerHa)) %>%
  tidyr::spread(Septoria , MedianYield)%>%
    mutate(YieldChange_Abs = Infected-Clean, 
           YieldChange_Rel= (YieldChange_Abs/Clean)*100) %>%
  ungroup() %>%
  group_by(Location, Climate) %>%
  summarise(Median_Diff = median(YieldChange_Rel),
            SD_Diff = sd(YieldChange_Rel))

work_data_summary