library(xlsx)
library(stargazer)

data = read.xlsx("result/回归数据2.xlsx", 1)
myfit = lm(log_death_rate~log_sum_info_value_mean+second_industry_ratio+log_hospital_beds+log_population_x+log_GDP+log_green_ground+log_gov_general_spend+log_industry_company+log_income+pm25, data)
myfit2 = lm(log_death_rate~log_sum_info_value_mean, data)
stargazer(myfit2, myfit,title = "results", align = F, type = "text", no.space = TRUE, out = "fit.html")