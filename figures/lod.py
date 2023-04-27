import numpy as np
import matplotlib.pyplot as plt


names = ['Br62', 'Br66', 'Cer77', 'Cer81', 'Colo45', 'Colo', 'Ov33', 'Ov79']
lods = np.asarray([
    [4.296050000000001, 100.0, 3.67038, 0.08924, 0.7212644999999999, 0.6605508, 1.69556, 17.571189999999998],
    [0.12497600000000002, 0.04426200000000001, 0.7218414, 0.205252, 0.08904500000000001, 0.042343000000000006, 0.321264, 0.803005]
])
corrs = np.asarray([
    [0.9976389683957072, 0.8907655469610752, 0.9956028799743847, 0.9961655281186113, 0.9996347212151226, 0.9698988793940768, 0.8843076623349918, 0.9949815019632563],
    [0.9991760767234513, 0.9903630626181422, 0.9997337431680235, 0.995348420770466, 0.9996855975450906, 0.9972440672840781, 0.8882889983537899, 0.9960687638236054],
])

plt.figure(figsize=(16, 8))

ax = plt.subplot(2, 1, 1)
xs = np.arange(8)
plt.bar(xs - 0.1, lods[0, :], color='steelblue', alpha=0.6, width=0.2, label='Default HPs')
plt.bar(xs + 0.1, lods[1, :], color='darkslateblue', alpha=0.6, width=0.2, label='Optimal HPs')
plt.xticks(xs, names)
plt.yscale('log')
plt.ylabel('Limit of detection')
plt.axvline(x=3.5, color='black', linewidth=1.0, linestyle='--', alpha=0.6)
plt.legend()
ax.spines[['right', 'top']].set_visible(False)

ax = plt.subplot(2, 1, 2)
xs = np.arange(8)
plt.bar(xs - 0.1, 1 - corrs[0, :], color='steelblue', alpha=0.6, width=0.2)
plt.bar(xs + 0.1, 1 - corrs[1, :], color='darkslateblue', alpha=0.6, width=0.2)
plt.xticks(xs, names)
plt.yscale('log')
plt.ylabel('1 - Pearson correlation')
plt.axvline(x=3.5, color='black', linewidth=1.0, linestyle='--', alpha=0.6)
ax.spines[['right', 'top']].set_visible(False)

plt.show()
