import k2cbv

campaign = 1
module = 18
model = 'everest1'
max_stars = 150

# Download the light curves
k2cbv.GetStars(campaign, module, model = model, max_stars = max_stars)

# Construct the design matrix
k2cbv.GetX(campaign, module, model = model, max_stars = max_stars)

# Fit all the stars
k2cbv.FitAll(campaign, module, model = model, max_stars = max_stars)
