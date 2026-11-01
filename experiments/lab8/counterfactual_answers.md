# Ответы на пять контрфактических вопросов

## Q1

В E1 обе модели улучшили ECE, NLL и Brier; измеренного контрпримера нет. Это не доказывает невозможность: ECE учитывает только максимальную вероятность и усреднение по интервалам, NLL/Brier — весь вектор вероятностей (8.4)-(8.5). На val NLL не растёт, потому что минимизируется именно эта функция, и T=1 явно включена как запасной кандидат. На test такой гарантии нет.

RUN_ID: 73968c7a84a5394c6aa0-5b8b87df36624d71a8c08781804ed7fd, 0799d8eab079cab5e620-ec0903e00a2f4df899b1f979e3032c7c, d0b5e7ac72f8f48c1504-51e91fd78fec46129a5d576e5ed42ee8, abcfea8ffce6144fa5ed-bf052e6973a84716bbbc19b858a7429e

```json
[
  {
    "model": "cnn",
    "RUN_ID": "73968c7a84a5394c6aa0-5b8b87df36624d71a8c08781804ed7fd",
    "fit_RUN_ID": "d0b5e7ac72f8f48c1504-51e91fd78fec46129a5d576e5ed42ee8",
    "ECE_delta": -0.18642878973458768,
    "NLL_delta": -0.730550042170818,
    "Brier_delta": -0.08476913390791402,
    "Tstar": 2.9563778545817163,
    "measured_counterexample": false
  },
  {
    "model": "vit",
    "RUN_ID": "0799d8eab079cab5e620-ec0903e00a2f4df899b1f979e3032c7c",
    "fit_RUN_ID": "abcfea8ffce6144fa5ed-bf052e6973a84716bbbc19b858a7429e",
    "ECE_delta": -0.4005361665162812,
    "NLL_delta": -2.3905222246632984,
    "Brier_delta": -0.2786346037412013,
    "Tstar": 4.273312679066656,
    "measured_counterexample": false
  }
]
```

## Q2

T5 подбирается на occlusion5 val и лучше соответствует известному сдвигу, но перенос на чистые данные не гарантирован. Разности T5-Tstar приведены выше отдельно для ECE/NLL/Brier/accuracy/confidence; отрицательные ECE/NLL/Brier означают улучшение. Классы не меняются. Для заранее неизвестного сдвига такой val нет; результат не является универсальным калибратором.

RUN_ID: 304e1000042dd417354d-12481f07601c4e3a8fe2fbc8e95f122a, 73968c7a84a5394c6aa0-5b8b87df36624d71a8c08781804ed7fd, d916efcb86dbc37ed0fd-5ea6c8d7ae3a4713abdc9196f866be8f

```json
{
  "clean": {
    "Tstar": 2.9563778545817163,
    "T5": 6.108733967046077,
    "RUN_ID": "73968c7a84a5394c6aa0-5b8b87df36624d71a8c08781804ed7fd",
    "T5_minus_Tstar": {
      "ECE": 0.23709780031006528,
      "accuracy": 0.0,
      "nll": 0.25248982727639,
      "brier": 0.08802296814808075,
      "mean_confidence": -0.2828077415022466
    },
    "all_metrics": [
      {
        "accuracy": 0.7074,
        "mean_confidence": 0.9222463045088207,
        "ece": {
          "value": 0.2148463045088206,
          "M": 15,
          "binning": "equal_width",
          "empty_bins": 3,
          "bins": [
            {
              "bin": 0,
              "count": 0,
              "accuracy": null,
              "confidence": null
            },
            {
              "bin": 1,
              "count": 0,
              "accuracy": null,
              "confidence": null
            },
            {
              "bin": 2,
              "count": 0,
              "accuracy": null,
              "confidence": null
            },
            {
              "bin": 3,
              "count": 1,
              "accuracy": 0.0,
              "confidence": 0.26350284338658003
            },
            {
              "bin": 4,
              "count": 24,
              "accuracy": 0.08333333333333333,
              "confidence": 0.2986063690811385
            },
            {
              "bin": 5,
              "count": 46,
              "accuracy": 0.21739130434782608,
              "confidence": 0.3697714865812404
            },
            {
              "bin": 6,
              "count": 109,
              "accuracy": 0.28440366972477066,
              "confidence": 0.4374809144039725
            },
            {
              "bin": 7,
              "count": 234,
              "accuracy": 0.36324786324786323,
              "confidence": 0.5042844414002451
            },
            {
              "bin": 8,
              "count": 262,
              "accuracy": 0.3969465648854962,
              "confidence": 0.5665595301422713
            },
            {
              "bin": 9,
              "count": 245,
              "accuracy": 0.3836734693877551,
              "confidence": 0.6333581300055952
            },
            {
              "bin": 10,
              "count": 313,
              "accuracy": 0.34824281150159747,
              "confidence": 0.7018517343693884
            },
            {
              "bin": 11,
              "count": 338,
              "accuracy": 0.3905325443786982,
              "confidence": 0.7670482504349615
            },
            {
              "bin": 12,
              "count": 432,
              "accuracy": 0.4513888888888889,
              "confidence": 0.8337328865409324
            },
            {
              "bin": 13,
              "count": 608,
              "accuracy": 0.5361842105263158,
              "confidence": 0.9042350838510449
            },
            {
              "bin": 14,
              "count": 7388,
              "accuracy": 0.8102328099621007,
              "confidence": 0.993480609613704
            }
          ]
        },
        "nll": 1.6403149518263471,
        "brier": 0.49371196774852233,
        "N": 10000,
        "temperature": 1.0
      },
      {
        "accuracy": 0.7074,
        "mean_confidence": 0.7247716601863092,
        "ece": {
          "value": 0.02841751477423294,
          "M": 15,
          "binning": "equal_width",
          "empty_bins": 2,
          "bins": [
            {
              "bin": 0,
              "count": 0,
              "accuracy": null,
              "confidence": null
            },
            {
              "bin": 1,
              "count": 0,
              "accuracy": null,
              "confidence": null
            },
            {
              "bin": 2,
              "count": 20,
              "accuracy": 0.05,
              "confidence": 0.1797931305494032
            },
            {
              "bin": 3,
              "count": 193,
              "accuracy": 0.23316062176165803,
              "confidence": 0.2390071889941634
            },
            {
              "bin": 4,
              "count": 439,
              "accuracy": 0.3325740318906606,
              "confidence": 0.30321168675383964
            },
            {
              "bin": 5,
              "count": 604,
              "accuracy": 0.39403973509933776,
              "confidence": 0.3678066390694354
            },
            {
              "bin": 6,
              "count": 707,
              "accuracy": 0.4031117397454031,
              "confidence": 0.4340760336082854
            },
            {
              "bin": 7,
              "count": 665,
              "accuracy": 0.5398496240601504,
              "confidence": 0.5000084008684311
            },
            {
              "bin": 8,
              "count": 622,
              "accuracy": 0.5643086816720257,
              "confidence": 0.5679722920279587
            },
            {
              "bin": 9,
              "count": 602,
              "accuracy": 0.6079734219269103,
              "confidence": 0.6328067299409608
            },
            {
              "bin": 10,
              "count": 644,
              "accuracy": 0.6413043478260869,
              "confidence": 0.6999020389949102
            },
            {
              "bin": 11,
              "count": 717,
              "accuracy": 0.7168758716875872,
              "confidence": 0.7674234353534108
            },
            {
              "bin": 12,
              "count": 774,
              "accuracy": 0.7958656330749354,
              "confidence": 0.8343801860194768
            },
            {
              "bin": 13,
              "count": 1157,
              "accuracy": 0.8565254969749352,
              "confidence": 0.9016420286862892
            },
            {
              "bin": 14,
              "count": 2856,
              "accuracy": 0.9625350140056023,
              "confidence": 0.9730783769113142
            }
          ]
        },
        "nll": 0.9097649096555291,
        "brier": 0.4089428338406083,
        "N": 10000,
        "temperature": 2.9563778545817163
      },
      {
        "accuracy": 0.7074,
        "mean_confidence": 0.44196391868406265,
        "ece": {
          "value": 0.2655153150842982,
          "M": 15,
          "binning": "equal_width",
          "empty_bins": 1,
          "bins": [
            {
              "bin": 0,
              "count": 0,
              "accuracy": null,
              "confidence": null
            },
            {
              "bin": 1,
              "count": 3,
              "accuracy": 0.0,
              "confidence": 0.1320562806015302
            },
            {
              "bin": 2,
              "count": 549,
              "accuracy": 0.28233151183970856,
              "confidence": 0.17936835375855228
            },
            {
              "bin": 3,
              "count": 1572,
              "accuracy": 0.42048346055979646,
              "confidence": 0.23427320940611757
            },
            {
              "bin": 4,
              "count": 1503,
              "accuracy": 0.5635395874916833,
              "confidence": 0.29904750383523765
            },
            {
              "bin": 5,
              "count": 1274,
              "accuracy": 0.6562009419152276,
              "confidence": 0.3652950728894821
            },
            {
              "bin": 6,
              "count": 1000,
              "accuracy": 0.775,
              "confidence": 0.4325251867542786
            },
            {
              "bin": 7,
              "count": 920,
              "accuracy": 0.8434782608695652,
              "confidence": 0.4993926276462281
            },
            {
              "bin": 8,
              "count": 839,
              "accuracy": 0.8951132300357568,
              "confidence": 0.567375549671276
            },
            {
              "bin": 9,
              "count": 774,
              "accuracy": 0.9457364341085271,
              "confidence": 0.6333810670610406
            },
            {
              "bin": 10,
              "count": 682,
              "accuracy": 0.9780058651026393,
              "confidence": 0.6984566778372584
            },
            {
              "bin": 11,
              "count": 532,
              "accuracy": 0.9849624060150376,
              "confidence": 0.7648632426334097
            },
            {
              "bin": 12,
              "count": 283,
              "accuracy": 0.9929328621908127,
              "confidence": 0.8296607375481564
            },
            {
              "bin": 13,
              "count": 68,
              "accuracy": 1.0,
              "confidence": 0.8884125149797301
            },
            {
              "bin": 14,
              "count": 1,
              "accuracy": 1.0,
              "confidence": 0.9458100944093121
            }
          ]
        },
        "nll": 1.1622547369319192,
        "brier": 0.49696580198868906,
        "N": 10000,
        "temperature": 6.108733967046077
      }
    ]
  },
  "shift5": {
    "Tstar": 2.9563778545817163,
    "T5": 6.108733967046077,
    "RUN_ID": "d916efcb86dbc37ed0fd-5ea6c8d7ae3a4713abdc9196f866be8f",
    "T5_minus_Tstar": {
      "ECE": -0.139416405039314,
      "accuracy": 0.0,
      "nll": -0.15932010364865556,
      "brier": -0.04356248257699091,
      "mean_confidence": -0.1791607879685522
    },
    "all_metrics": [
      {
        "accuracy": 0.2447,
        "mean_confidence": 0.7527714883000732,
        "ece": {
          "value": 0.5080714883000732,
          "M": 15,
          "binning": "equal_width",
          "empty_bins": 2,
          "bins": [
            {
              "bin": 0,
              "count": 0,
              "accuracy": null,
              "confidence": null
            },
            {
              "bin": 1,
              "count": 0,
              "accuracy": null,
              "confidence": null
            },
            {
              "bin": 2,
              "count": 3,
              "accuracy": 0.0,
              "confidence": 0.190106781612857
            },
            {
              "bin": 3,
              "count": 36,
              "accuracy": 0.05555555555555555,
              "confidence": 0.24678693921045733
            },
            {
              "bin": 4,
              "count": 152,
              "accuracy": 0.14473684210526316,
              "confidence": 0.30716875395786564
            },
            {
              "bin": 5,
              "count": 393,
              "accuracy": 0.15012722646310434,
              "confidence": 0.37057192323847465
            },
            {
              "bin": 6,
              "count": 536,
              "accuracy": 0.11567164179104478,
              "confidence": 0.4355719268836701
            },
            {
              "bin": 7,
              "count": 773,
              "accuracy": 0.16170763260025872,
              "confidence": 0.5015946141157797
            },
            {
              "bin": 8,
              "count": 811,
              "accuracy": 0.17755856966707767,
              "confidence": 0.5663458864604727
            },
            {
              "bin": 9,
              "count": 812,
              "accuracy": 0.2044334975369458,
              "confidence": 0.6333211276427464
            },
            {
              "bin": 10,
              "count": 763,
              "accuracy": 0.23722149410222804,
              "confidence": 0.7009724545622522
            },
            {
              "bin": 11,
              "count": 835,
              "accuracy": 0.23353293413173654,
              "confidence": 0.7660546456675491
            },
            {
              "bin": 12,
              "count": 901,
              "accuracy": 0.195338512763596,
              "confidence": 0.8336784563940491
            },
            {
              "bin": 13,
              "count": 1194,
              "accuracy": 0.2554438860971524,
              "confidence": 0.9018004918757051
            },
            {
              "bin": 14,
              "count": 2791,
              "accuracy": 0.3618774632748119,
              "confidence": 0.977707513023816
            }
          ]
        },
        "nll": 4.31040198176631,
        "brier": 1.1903257792578836,
        "N": 10000,
        "temperature": 1.0
      },
      {
        "accuracy": 0.2447,
        "mean_confidence": 0.41328743372560167,
        "ece": {
          "value": 0.1685977962129352,
          "M": 15,
          "binning": "equal_width",
          "empty_bins": 2,
          "bins": [
            {
              "bin": 0,
              "count": 0,
              "accuracy": null,
              "confidence": null
            },
            {
              "bin": 1,
              "count": 0,
              "accuracy": null,
              "confidence": null
            },
            {
              "bin": 2,
              "count": 339,
              "accuracy": 0.14454277286135694,
              "confidence": 0.18193728140524765
            },
            {
              "bin": 3,
              "count": 1397,
              "accuracy": 0.13743736578382248,
              "confidence": 0.23751433985339274
            },
            {
              "bin": 4,
              "count": 2061,
              "accuracy": 0.16545366327025715,
              "confidence": 0.2998984122572618
            },
            {
              "bin": 5,
              "count": 1776,
              "accuracy": 0.22184684684684686,
              "confidence": 0.3661113739708572
            },
            {
              "bin": 6,
              "count": 1368,
              "accuracy": 0.25146198830409355,
              "confidence": 0.4323587954018393
            },
            {
              "bin": 7,
              "count": 1000,
              "accuracy": 0.271,
              "confidence": 0.4984707976199123
            },
            {
              "bin": 8,
              "count": 655,
              "accuracy": 0.3450381679389313,
              "confidence": 0.5651942643162826
            },
            {
              "bin": 9,
              "count": 507,
              "accuracy": 0.32938856015779094,
              "confidence": 0.6314564340826727
            },
            {
              "bin": 10,
              "count": 360,
              "accuracy": 0.39444444444444443,
              "confidence": 0.7002584414825427
            },
            {
              "bin": 11,
              "count": 254,
              "accuracy": 0.47244094488188976,
              "confidence": 0.7657923142470325
            },
            {
              "bin": 12,
              "count": 166,
              "accuracy": 0.6927710843373494,
              "confidence": 0.8333629324717373
            },
            {
              "bin": 13,
              "count": 95,
              "accuracy": 0.6842105263157895,
              "confidence": 0.8937540232334542
            },
            {
              "bin": 14,
              "count": 22,
              "accuracy": 0.9545454545454546,
              "confidence": 0.9521903437878371
            }
          ]
        },
        "nll": 2.3024917609665603,
        "brier": 0.9007746564044231,
        "N": 10000,
        "temperature": 2.9563778545817163
      },
      {
        "accuracy": 0.2447,
        "mean_confidence": 0.23412664575704947,
        "ece": {
          "value": 0.02918139117362119,
          "M": 15,
          "binning": "equal_width",
          "empty_bins": 5,
          "bins": [
            {
              "bin": 0,
              "count": 0,
              "accuracy": null,
              "confidence": null
            },
            {
              "bin": 1,
              "count": 58,
              "accuracy": 0.1206896551724138,
              "confidence": 0.12757307498020867
            },
            {
              "bin": 2,
              "count": 3628,
              "accuracy": 0.14856670341786107,
              "confidence": 0.17410169413023743
            },
            {
              "bin": 3,
              "count": 3868,
              "accuracy": 0.23474663908996898,
              "confidence": 0.22968510486066113
            },
            {
              "bin": 4,
              "count": 1549,
              "accuracy": 0.33892834086507423,
              "confidence": 0.29467431746938905
            },
            {
              "bin": 5,
              "count": 565,
              "accuracy": 0.4194690265486726,
              "confidence": 0.3614742498368916
            },
            {
              "bin": 6,
              "count": 215,
              "accuracy": 0.6697674418604651,
              "confidence": 0.429561312763216
            },
            {
              "bin": 7,
              "count": 86,
              "accuracy": 0.6744186046511628,
              "confidence": 0.49541930993188293
            },
            {
              "bin": 8,
              "count": 21,
              "accuracy": 0.9523809523809523,
              "confidence": 0.5615591376752077
            },
            {
              "bin": 9,
              "count": 8,
              "accuracy": 0.875,
              "confidence": 0.6235485157576086
            },
            {
              "bin": 10,
              "count": 2,
              "accuracy": 1.0,
              "confidence": 0.6889727413517339
            },
            {
              "bin": 11,
              "count": 0,
              "accuracy": null,
              "confidence": null
            },
            {
              "bin": 12,
              "count": 0,
              "accuracy": null,
              "confidence": null
            },
            {
              "bin": 13,
              "count": 0,
              "accuracy": null,
              "confidence": null
            },
            {
              "bin": 14,
              "count": 0,
              "accuracy": null,
              "confidence": null
            }
          ]
        },
        "nll": 2.1431716573179047,
        "brier": 0.8572121738274322,
        "N": 10000,
        "temperature": 6.108733967046077
      }
    ]
  }
}
```

## Q3

Предсказанный заранее порядок проверен на одинаковых test ID. При одинаковом ограничении Linf случайный знак не согласован с градиентом CE; FGSM максимизирует линейное приближение, PGD многократно уточняет направление с проекцией (8.9)-(8.10). Наблюдение относится к этим двум замороженным моделям и epsilon=2/255, а не ко всем возможным шумам.

RUN_ID: 07fec26d48c701023bbd-28ec2dc11eff416185d2bb75ca985d2f, 1a44bb9104623fa07cb4-60472984ecd94626b9d910b76e11e7c3, 00aed6b3019eda8bd2b0-e1833f1367e4430f8aeccd94a49d11b6, 4ed189b2245186f3b436-7011dd2f77284357931c0e901a28d9d2, be749c9af8d7ccb534e2-69374eeafcc54854810477dd4e5d9500, 79a1b4800fce5956a2d9-2d085f1bce3b439aa37201e020dabd75

```json
[
  {
    "model": "cnn",
    "random_accuracy": 0.7057,
    "FGSM_accuracy": 0.2988,
    "PGD10_accuracy": 0.2458,
    "prediction_random_gt_FGSM_ge_PGD": true,
    "RUN_IDs": [
      "07fec26d48c701023bbd-28ec2dc11eff416185d2bb75ca985d2f",
      "1a44bb9104623fa07cb4-60472984ecd94626b9d910b76e11e7c3",
      "00aed6b3019eda8bd2b0-e1833f1367e4430f8aeccd94a49d11b6"
    ],
    "epsilon": 0.00784313725490196
  },
  {
    "model": "vit",
    "random_accuracy": 0.4489,
    "FGSM_accuracy": 0.1031,
    "PGD10_accuracy": 0.076,
    "prediction_random_gt_FGSM_ge_PGD": true,
    "RUN_IDs": [
      "4ed189b2245186f3b436-7011dd2f77284357931c0e901a28d9d2",
      "be749c9af8d7ccb534e2-69374eeafcc54854810477dd4e5d9500",
      "79a1b4800fce5956a2d9-2d085f1bce3b439aa37201e020dabd75"
    ],
    "epsilon": 0.00784313725490196
  }
]
```

## Q4

Температура изменяет уверенность и ECE, но не исправляет ни одного ошибочного класса: число изменившихся предсказаний и ошибок равно нулю (8.6). Подбор на чистой val не гарантирует калибровку атакованных объектов. Метрики вычислены по одним и тем же сохранённым FGSM logits.

RUN_ID: 1a44bb9104623fa07cb4-60472984ecd94626b9d910b76e11e7c3, be749c9af8d7ccb534e2-69374eeafcc54854810477dd4e5d9500

```json
[
  {
    "model": "cnn",
    "RUN_ID": "1a44bb9104623fa07cb4-60472984ecd94626b9d910b76e11e7c3",
    "T1": {
      "accuracy": 0.2988,
      "mean_confidence": 0.9246208368362175,
      "ece": {
        "value": 0.6259210347032519,
        "M": 15,
        "binning": "equal_width",
        "empty_bins": 3,
        "bins": [
          {
            "bin": 0,
            "count": 0,
            "accuracy": null,
            "confidence": null
          },
          {
            "bin": 1,
            "count": 0,
            "accuracy": null,
            "confidence": null
          },
          {
            "bin": 2,
            "count": 0,
            "accuracy": null,
            "confidence": null
          },
          {
            "bin": 3,
            "count": 2,
            "accuracy": 0.5,
            "confidence": 0.24950533241402892
          },
          {
            "bin": 4,
            "count": 10,
            "accuracy": 0.2,
            "confidence": 0.304472230119344
          },
          {
            "bin": 5,
            "count": 33,
            "accuracy": 0.3333333333333333,
            "confidence": 0.3749247685636508
          },
          {
            "bin": 6,
            "count": 88,
            "accuracy": 0.2840909090909091,
            "confidence": 0.43761565540583436
          },
          {
            "bin": 7,
            "count": 212,
            "accuracy": 0.23113207547169812,
            "confidence": 0.5052915239633157
          },
          {
            "bin": 8,
            "count": 291,
            "accuracy": 0.30584192439862545,
            "confidence": 0.5657127839447694
          },
          {
            "bin": 9,
            "count": 260,
            "accuracy": 0.2923076923076923,
            "confidence": 0.6341072321113403
          },
          {
            "bin": 10,
            "count": 305,
            "accuracy": 0.2754098360655738,
            "confidence": 0.6992660374941092
          },
          {
            "bin": 11,
            "count": 308,
            "accuracy": 0.3538961038961039,
            "confidence": 0.7694536919924648
          },
          {
            "bin": 12,
            "count": 413,
            "accuracy": 0.3196125907990315,
            "confidence": 0.8356065460569625
          },
          {
            "bin": 13,
            "count": 658,
            "accuracy": 0.3161094224924012,
            "confidence": 0.9023715818931043
          },
          {
            "bin": 14,
            "count": 7420,
            "accuracy": 0.2967654986522911,
            "confidence": 0.9927272174965188
          }
        ]
      },
      "nll": 6.54839871810891,
      "brier": 1.297093939017459,
      "N": 10000,
      "temperature": 1.0
    },
    "Tstar": {
      "accuracy": 0.2988,
      "mean_confidence": 0.7170806744536737,
      "ece": {
        "value": 0.41846040351261626,
        "M": 15,
        "binning": "equal_width",
        "empty_bins": 2,
        "bins": [
          {
            "bin": 0,
            "count": 0,
            "accuracy": null,
            "confidence": null
          },
          {
            "bin": 1,
            "count": 0,
            "accuracy": null,
            "confidence": null
          },
          {
            "bin": 2,
            "count": 6,
            "accuracy": 0.3333333333333333,
            "confidence": 0.1835591175479121
          },
          {
            "bin": 3,
            "count": 138,
            "accuracy": 0.21014492753623187,
            "confidence": 0.24012242572665043
          },
          {
            "bin": 4,
            "count": 348,
            "accuracy": 0.21551724137931033,
            "confidence": 0.30312843495793385
          },
          {
            "bin": 5,
            "count": 498,
            "accuracy": 0.2751004016064257,
            "confidence": 0.36839734916619954
          },
          {
            "bin": 6,
            "count": 696,
            "accuracy": 0.3074712643678161,
            "confidence": 0.432453154438588
          },
          {
            "bin": 7,
            "count": 714,
            "accuracy": 0.3291316526610644,
            "confidence": 0.5006224669339209
          },
          {
            "bin": 8,
            "count": 727,
            "accuracy": 0.2984869325997249,
            "confidence": 0.56657188764664
          },
          {
            "bin": 9,
            "count": 775,
            "accuracy": 0.31225806451612903,
            "confidence": 0.6333722041183368
          },
          {
            "bin": 10,
            "count": 769,
            "accuracy": 0.30299089726918077,
            "confidence": 0.7011684844220509
          },
          {
            "bin": 11,
            "count": 904,
            "accuracy": 0.32079646017699115,
            "confidence": 0.7671647023613565
          },
          {
            "bin": 12,
            "count": 1038,
            "accuracy": 0.28998073217726394,
            "confidence": 0.8348766704360522
          },
          {
            "bin": 13,
            "count": 1382,
            "accuracy": 0.28364688856729375,
            "confidence": 0.9019197528888718
          },
          {
            "bin": 14,
            "count": 2005,
            "accuracy": 0.30972568578553616,
            "confidence": 0.9679073938054276
          }
        ]
      },
      "nll": 2.571087587051574,
      "brier": 1.0576379741776134,
      "N": 10000,
      "temperature": 2.9563778545817163
    },
    "prediction_changes": 0,
    "error_changes": 0
  },
  {
    "model": "vit",
    "RUN_ID": "be749c9af8d7ccb534e2-69374eeafcc54854810477dd4e5d9500",
    "T1": {
      "accuracy": 0.1031,
      "mean_confidence": 0.9611793317381067,
      "ece": {
        "value": 0.8580793317381069,
        "M": 15,
        "binning": "equal_width",
        "empty_bins": 4,
        "bins": [
          {
            "bin": 0,
            "count": 0,
            "accuracy": null,
            "confidence": null
          },
          {
            "bin": 1,
            "count": 0,
            "accuracy": null,
            "confidence": null
          },
          {
            "bin": 2,
            "count": 0,
            "accuracy": null,
            "confidence": null
          },
          {
            "bin": 3,
            "count": 0,
            "accuracy": null,
            "confidence": null
          },
          {
            "bin": 4,
            "count": 1,
            "accuracy": 0.0,
            "confidence": 0.30011976250359373
          },
          {
            "bin": 5,
            "count": 9,
            "accuracy": 0.1111111111111111,
            "confidence": 0.3802770656956407
          },
          {
            "bin": 6,
            "count": 29,
            "accuracy": 0.10344827586206896,
            "confidence": 0.4375774138729381
          },
          {
            "bin": 7,
            "count": 97,
            "accuracy": 0.20618556701030927,
            "confidence": 0.5076093026450258
          },
          {
            "bin": 8,
            "count": 126,
            "accuracy": 0.18253968253968253,
            "confidence": 0.5655756281165278
          },
          {
            "bin": 9,
            "count": 142,
            "accuracy": 0.18309859154929578,
            "confidence": 0.6324256380796217
          },
          {
            "bin": 10,
            "count": 153,
            "accuracy": 0.23529411764705882,
            "confidence": 0.7018470274722891
          },
          {
            "bin": 11,
            "count": 181,
            "accuracy": 0.2541436464088398,
            "confidence": 0.7671593576341135
          },
          {
            "bin": 12,
            "count": 285,
            "accuracy": 0.21052631578947367,
            "confidence": 0.8358120896428218
          },
          {
            "bin": 13,
            "count": 378,
            "accuracy": 0.17195767195767195,
            "confidence": 0.9014432719223601
          },
          {
            "bin": 14,
            "count": 8599,
            "accuracy": 0.08733573671357135,
            "confidence": 0.9954512670832123
          }
        ]
      },
      "nll": 9.469269377407116,
      "brier": 1.7323673696842559,
      "N": 10000,
      "temperature": 1.0
    },
    "Tstar": {
      "accuracy": 0.1031,
      "mean_confidence": 0.572030109350508,
      "ece": {
        "value": 0.46893010935050816,
        "M": 15,
        "binning": "equal_width",
        "empty_bins": 5,
        "bins": [
          {
            "bin": 0,
            "count": 0,
            "accuracy": null,
            "confidence": null
          },
          {
            "bin": 1,
            "count": 0,
            "accuracy": null,
            "confidence": null
          },
          {
            "bin": 2,
            "count": 6,
            "accuracy": 0.0,
            "confidence": 0.19021236172909586
          },
          {
            "bin": 3,
            "count": 120,
            "accuracy": 0.125,
            "confidence": 0.24291288015744258
          },
          {
            "bin": 4,
            "count": 396,
            "accuracy": 0.1414141414141414,
            "confidence": 0.30483592034471574
          },
          {
            "bin": 5,
            "count": 726,
            "accuracy": 0.19146005509641872,
            "confidence": 0.3677176035189323
          },
          {
            "bin": 6,
            "count": 875,
            "accuracy": 0.1977142857142857,
            "confidence": 0.43317288793898334
          },
          {
            "bin": 7,
            "count": 1031,
            "accuracy": 0.18331716779825413,
            "confidence": 0.5014399014695383
          },
          {
            "bin": 8,
            "count": 1484,
            "accuracy": 0.12196765498652291,
            "confidence": 0.5705732564049533
          },
          {
            "bin": 9,
            "count": 2624,
            "accuracy": 0.07507621951219512,
            "confidence": 0.6378408046184202
          },
          {
            "bin": 10,
            "count": 2737,
            "accuracy": 0.02959444647424187,
            "confidence": 0.6887696598573829
          },
          {
            "bin": 11,
            "count": 1,
            "accuracy": 0.0,
            "confidence": 0.733910889514255
          },
          {
            "bin": 12,
            "count": 0,
            "accuracy": null,
            "confidence": null
          },
          {
            "bin": 13,
            "count": 0,
            "accuracy": null,
            "confidence": null
          },
          {
            "bin": 14,
            "count": 0,
            "accuracy": null,
            "confidence": null
          }
        ]
      },
      "nll": 2.790910680682641,
      "brier": 1.1574109908107069,
      "N": 10000,
      "temperature": 4.273312679066656
    },
    "prediction_changes": 0,
    "error_changes": 0
  }
]
```

## Q5

Температура сохраняет порядок логитов внутри объекта, но не обязана сохранять порядок максимальных вероятностей между объектами в многоклассовой задаче. Поэтому AURC и OOD AUROC могут измениться; измеренные значения и число изменённых позиций указаны выше. Инвариантность AUROC касается строго возрастающего преобразования уже готовой скалярной U (8.13), а softmax(z/T) в общем случае не является одним таким преобразованием MSP для всех векторов z.

RUN_ID: c4bd07267e884795ad42-9f79f9e6c4534bf9ae92ce4d5fbd849f, 28c6a867fdd60b50e742-b8ff8463a4ac44f38517e4b96fe64ebe

```json
[
  {
    "model": "cnn",
    "RUN_ID": "c4bd07267e884795ad42-9f79f9e6c4534bf9ae92ce4d5fbd849f",
    "ID_RUN_ID": "73968c7a84a5394c6aa0-5b8b87df36624d71a8c08781804ed7fd",
    "OOD_RUN_ID": "34794323dd9b17b39aab-ce0f18c4ab574e9cbef3ca9b897b3573",
    "T1": {
      "AURC": 0.11593428145562912,
      "AUROC": 0.6740990550092194
    },
    "Tstar": {
      "temperature": 2.9563778545817163,
      "AURC": 0.11497476354919964,
      "AUROC": 0.6673059081130915
    },
    "ID_rank_positions_changed": 9973,
    "class_changes": 0
  },
  {
    "model": "vit",
    "RUN_ID": "28c6a867fdd60b50e742-b8ff8463a4ac44f38517e4b96fe64ebe",
    "ID_RUN_ID": "0799d8eab079cab5e620-ec0903e00a2f4df899b1f979e3032c7c",
    "OOD_RUN_ID": "52d6409989c08642ae8b-aacd69d79cb84b059ae14c48f4f859fa",
    "T1": {
      "AURC": 0.3930638461251112,
      "AUROC": 0.578932786570375
    },
    "Tstar": {
      "temperature": 4.273312679066656,
      "AURC": 0.3850406258742513,
      "AUROC": 0.592006069452981
    },
    "ID_rank_positions_changed": 9981,
    "class_changes": 0
  }
]
```
