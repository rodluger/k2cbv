import k2cbv

campaign = 4
module = 15
model = 'everest1'
max_stars = 300

# Download the light curves
time, breakpoints, fluxes = k2cbv.GetStars(campaign, module, model = model, max_stars = max_stars)

# Construct the design matrix
X = k2cbv.GetX(campaign, module, model = model, max_stars = max_stars)

# Fit all the stars
k2cbv.FitAll(campaign, module, model = model, max_stars = max_stars)