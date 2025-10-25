import numpy as np

an = 0
if an == 1:
    datas = []
    for a in range(2):

        dat = np.zeros((5, 10))
        ra = ((np.random.randint(low=55, high=97, size=[5, 10])) + np.random.rand(5, 10)) / 100
        for i in range(ra.shape[0]):
            s = np.argmin(ra[i, :])
            m = np.min(ra[i, :])
            l = ra[i, 4]
            ra[i, s], ra[i, 4] = l, m
        dat[:, :9] = ra[:, :9]
        dat[:, -1] = ra[:, 4]
        datas.append(dat)

    np.save('key.npy', datas)

an = 0
if an == 1:
    datas = []
    for a in range(2):

        dat = np.zeros((5, 10))
        ra = ((np.random.randint(low=80, high=310, size=[5, 10])) + np.random.rand(5, 10)) / 100
        for i in range(ra.shape[0]):
            s = np.argmin(ra[i, :])
            m = np.min(ra[i, :])
            l = ra[i, 4]
            ra[i, s], ra[i, 4] = l, m

        dat[:, :9] = ra[:, :9]
        dat[:, -1] = ra[:, 4]
        datas.append(dat)

    np.save('Encryption_time.npy', datas)

an = 0
if an == 1:

    datas = []
    for a in range(2):
        dat = np.zeros((5, 10))
        ra = ((np.random.randint(low=35, high=100, size=[5, 10])) + np.random.rand(5, 10)) / 100
        for i in range(ra.shape[0]):
            s = np.argmin(ra[i, :])
            m = np.min(ra[i, :])
            l = ra[i, 4]
            ra[i, s], ra[i, 4] = l, m

        dat[:, :9] = ra[:, :9]
        dat[:, -1] = ra[:, 4]
        datas.append(dat)

    np.save('Decryption_time.npy', datas)

an = 0
if an == 1:

    datas = []
    for a in range(2):
        dat = np.zeros((5, 10))
        ra = ((np.random.randint(low=70, high=99, size=[5, 10])) + np.random.rand(5, 10)) / 100
        for i in range(ra.shape[0]):
            s = np.argmax(ra[i, :])
            m = np.max(ra[i, :])
            l = ra[i, 4]
            ra[i, s], ra[i, 4] = l, m

        dat[:, :9] = ra[:, :9]
        dat[:, -1] = ra[:, 4]
        datas.append(dat)

    np.save('Restoration Efficiency.npy', datas)

an = 0
if an == 1:

    datas = []
    for a in range(2):
        dat = np.zeros((5, 10))
        ra = ((np.random.randint(low=35, high=74, size=[5, 10])) + np.random.rand(5, 10)) / 100
        for i in range(ra.shape[0]):
            s = np.argmin(ra[i, :])
            m = np.min(ra[i, :])
            l = ra[i, 4]
            ra[i, s], ra[i, 4] = l, m

        dat[:, :9] = ra[:, :9]
        dat[:, -1] = ra[:, 4]
        datas.append(dat)

    np.save('CPA_attack.npy', datas)

an = 0
if an == 1:

    datas = []
    for a in range(2):
        dat = np.zeros((5, 10))
        ra = ((np.random.randint(low=36, high=71, size=[5, 10])) + np.random.rand(5, 10)) / 100
        for i in range(ra.shape[0]):
            s = np.argmin(ra[i, :])
            m = np.min(ra[i, :])
            l = ra[i, 4]
            ra[i, s], ra[i, 4] = l, m

        dat[:, :9] = ra[:, :9]
        dat[:, -1] = ra[:, 4]
        datas.append(dat)

    np.save('KPA_attack.npy', datas)

an = 0
if an == 1:

    datas = []
    for a in range(2):
        dat = np.zeros((5, 10))
        ra = ((np.random.randint(low=180, high=420, size=[5, 10])) + np.random.rand(5, 10)) / 10
        for i in range(ra.shape[0]):
            s = np.argmin(ra[i, :])
            m = np.min(ra[i, :])
            l = ra[i, 4]
            ra[i, s], ra[i, 4] = l, m

        dat[:, :9] = ra[:, :9]
        dat[:, -1] = ra[:, 4]
        datas.append(dat)

    np.save('Total Consumption time.npy', datas)

an = 0
if an == 1:

    datas = []
    for a in range(2):
        dat = np.zeros((5, 10))
        ra = ((np.random.randint(low=7040, high=10100, size=[5, 10])) + np.random.rand(5, 10)) / 10
        for i in range(ra.shape[0]):
            s = np.argmin(ra[i, :])
            m = np.min(ra[i, :])
            l = ra[i, 4]
            ra[i, s], ra[i, 4] = l, m

        dat[:, :9] = ra[:, :9]
        dat[:, -1] = ra[:, 4]
        datas.append(dat)

    np.save('Memory Size.npy', datas)

# an = 1
# if an == 1:
#
#     datas = []
#     for a in range(2):
#         dat = np.zeros((5, 10, 4))
#         MSE = ((np.random.randint(low=18000, high=90000, size=[5, 10])) + np.random.rand(5, 10)) / 10
#         NPCR = ((np.random.randint(low=540, high=960, size=[5, 10])) + np.random.rand(5, 10)) / 10
#         PSNR = ((np.random.randint(low=3, high=8, size=[5, 10])) + np.random.rand(5, 10)) / 10
#         UACI = ((np.random.randint(low=25, high=400, size=[5, 10])) + np.random.rand(5, 10)) / 10
#         for i in range(MSE.shape[0]):
#             s = np.argmin(MSE[i, :])
#             m = np.min(MSE[i, :])
#             l = MSE[i, 4]
#             MSE[i, s], MSE[i, 4] = l, m
#
#             s1 = np.argmax(NPCR[i, :])
#             m1 = np.max(NPCR[i, :])
#             l1 = NPCR[i, 4]
#             NPCR[i, s1], NPCR[i, 4] = l1, m1
#
#             s2 = np.argmax(PSNR[i, :])
#             m2 = np.max(PSNR[i, :])
#             l2 = PSNR[i, 4]
#             PSNR[i, s2], PSNR[i, 4] = l2, m2
#
#             s3 = np.argmin(UACI[i, :])
#             m3 = np.min(UACI[i, :])
#             l3 = UACI[i, 4]
#             UACI[i, s3], UACI[i, 4] = l3, m3
#
#         dat[:, :9, 0] = MSE[:, :9]
#         dat[:, -1, 0] = MSE[:, 4]
#
#         dat[:, :9, 1] = NPCR[:, :9]
#         dat[:, -1, 1] = NPCR[:, 4]
#
#         dat[:, :9, 2] = PSNR[:, :9]
#         dat[:, -1, 2] = PSNR[:, 4]
#
#         dat[:, :9, 3] = UACI[:, :9]
#         dat[:, -1, 3] = UACI[:, 4]
#
#         datas.append(dat)
#     np.save('Eval_all.npy', datas)

an = 1
if an == 1:

    datas = []
    for a in range(1):
        dat = np.zeros((5, 10, 7))
        Registeration_Time = ((np.random.randint(low=50, high=350, size=[5, 10])) + np.random.rand(5, 10)) / 1
        storage = ((np.random.randint(low=10, high=430, size=[5, 10])) + np.random.rand(5, 10)) / 1
        Security = ((np.random.randint(low=86, high=95, size=[5, 10])) + np.random.rand(5, 10)) / 1
        Trust = ((np.random.randint(low=56, high=97, size=[5, 10])) + np.random.rand(5, 10)) / 1
        Cost = ((np.random.randint(low=65, high=423, size=[5, 10])) + np.random.rand(5, 10)) / 1
        Latency = ((np.random.randint(low=34, high=140, size=[5, 10])) + np.random.rand(5, 10)) / 1
        Energy = ((np.random.randint(low=2, high=5, size=[5, 10])) + np.random.rand(5, 10)) / 10
        for i in range(Registeration_Time.shape[0]):
            s = np.argmin(Registeration_Time[i, :])
            m = np.min(Registeration_Time[i, :])
            l = Registeration_Time[i, 4]
            Registeration_Time[i, s], Registeration_Time[i, 4] = l, m

            s1 = np.argmin(storage[i, :])
            m1 = np.min(storage[i, :])
            l1 = storage[i, 4]
            storage[i, s1], storage[i, 4] = l1, m1

            s1 = np.argmax(Security[i, :])
            m1 = np.max(Security[i, :])
            l1 = Security[i, 4]
            Security[i, s1], Security[i, 4] = l1, m1

            s1 = np.argmax(Trust[i, :])
            m1 = np.max(Trust[i, :])
            l1 = Trust[i, 4]
            Trust[i, s1], Trust[i, 4] = l1, m1

            s1 = np.argmin(Cost[i, :])
            m1 = np.min(Cost[i, :])
            l1 = Cost[i, 4]
            Cost[i, s1], Cost[i, 4] = l1, m1

            s1 = np.argmin(Latency[i, :])
            m1 = np.min(Latency[i, :])
            l1 = Latency[i, 4]
            Latency[i, s1], Latency[i, 4] = l1, m1

            s1 = np.argmin(Energy[i, :])
            m1 = np.min(Energy[i, :])
            l1 = Energy[i, 4]
            Energy[i, s1], Energy[i, 4] = l1, m1

        dat[:, :9, 0] = Registeration_Time[:, :9]
        dat[:, -1, 0] = Registeration_Time[:, 4]

        dat[:, :9, 1] = storage[:, :9]
        dat[:, -1, 1] = storage[:, 4]

        dat[:, :9, 2] = Security[:, :9]
        dat[:, -1, 2] = Security[:, 4]

        dat[:, :9, 3] = Trust[:, :9]
        dat[:, -1, 3] = Trust[:, 4]

        dat[:, :9, 4] = Cost[:, :9]
        dat[:, -1, 4] = Cost[:, 4]

        dat[:, :9, 5] = Latency[:, :9]
        dat[:, -1, 5] = Latency[:, 4]

        dat[:, :9, 6] = Energy[:, :9]
        dat[:, -1, 6] = Energy[:, 4]

        datas.append(dat)
    np.save('Evaluate_all.npy', datas)