# Hand Rub Tensor Parallelism

## Many accounts can be settled clearly.

Recently, my work and life have encountered bottlenecks.

In life, due to the turmoil of the current situation, I once again have the feeling of being trapped in a garbage moment in history. So these days, I started reading Camus’ biography and “The Plague”.

<details>
<summary>An excerpt of his evaluation of Camus</summary>

> In the face of ubiquitous absurdity, it is not enough to look on or laugh at it. If the absurd life and world are a shadow, then the discussion of absurdity itself will eventually lead us back to the sunshine. Therefore, Camus's philosophy of resistance against suicide is precious in any era and has become an immortal legacy he left to future generations. Furthermore, finding eternity in absurdity, generating meaning from nothingness, borrowing inspiration from boring places, and asking whether there is innovation in leisure time are the real issues throughout the life of modern people.
>
> --- [110th Anniversary of Camus’s Birth | Finding Meaning from Absurdity](https://www.chinawriter.com.cn/n1/2023/1108/c404091-40113499.html)
</details>

At work, in order to truly complete the RL of ultra-large-scale MOE models such as DeepSeek in the open source community, we have stepped on a lot of pitfalls. The specific troubles will always be told elsewhere one day, but there is one sentence that is worth sharing immediately.

> When everyone develops RL systems, they always write too much code and do too little calculations.

This is the idea given by a senior after reading the working documents of our MOE RL. I was deeply shocked after hearing this. This sentence awakened our misunderstanding. With such high-intensity development nowadays, it seems inevitable to fix countless bugs with mem-savor and megatron resharder. In fact, most of them can be solved by first theoretically calculating the upper bound and then redesigning the system. This reminds me of something else that happened the other day:

1. A friend finally left the United States and returned to China to work after working in the Bay Area linen industry for several years. I asked him whether "Bay Area Linen" was wrong, or "Bay Area" was wrong, or "linen" was wrong, or both "Bay Area" and "flax" were wrong. He told me that both "Bay Area" and "Flax" were wrong, so he eliminated the two wrong options. In my opinion, if we pause the busy development at hand for a while and think about it carefully, maybe our work can also eliminate two wrong options at once.
2. I don’t remember who told me that inferior engineers can’t complete the requirements, medium engineers can complete a large number of requirements, and top engineers will make everyone’s requirements simpler. We have to stop now and make the requirements simple.

Whatever, in fact, in the RL system, many things can be calculated. Although actual measurement is important, calculating clearly means that you have a very clear understanding of the system. There is no reason not to do some basic calculations before actually doing the experiment, get basic expectations, and then use the expectations to guide the experiment. Even when we finally write and submit the manuscript, we may not write the preliminary calculations into the article at all, but such calculations let us know that the current development and experiments are on the right path. Thinking about it this way, the deployment costs mentioned in the tech report of DeepSeek V3 were also completed in engineering practice under the guidance of theoretical calculations. Recently, SGLang has become popular [96 H100 reproduces DeepSeek official inference cost](https://lmsys.org/blog/2025-05-05-large-scale-ep/). Various parameters, including the 96 card itself, must be calculated first and then put into practice.All in all, there are bottlenecks in life and work anyway, so I plan to stop for a few days to figure out the accounts before continuing development. This note is about TP and lays the foundation for our subsequent deduction of DeepSeek’s GRPO cost.

## Segmentation of TP

Just like the linear algebra learned in freshman year, for matrix multiplication, we can split it into the multiplication of several multi-submatrices.

$$
A \times B = \begin{bmatrix}
A_{11} & A_{12} \\
A_{21} & A_{22}
\end{bmatrix} \times \begin{bmatrix}
B_{11} & B_{12} \\
B_{21} & B_{22}
\end{bmatrix} = \begin{bmatrix}
A_{11}B_{11} + A_{12}B_{21} & A_{11}B_{12} + A_{12}B_{22} \\
A_{21}B_{11} + A_{22}B_{21} & A_{21}B_{12} + A_{22}B_{22}
\end{bmatrix}
$$

But the truth is, it's very confusing to split it up like this. When a single GPU really cannot fit matrix B, there is no need to cut B into four pieces. The main way to split is to split along rows or split along columns.

1. Split B along the rows, then A also needs to be split along the columns.

$$
A \times B = \begin{bmatrix}
A_1 & A_2
\end{bmatrix} \times \begin{bmatrix}
B_1 \\
\\
B_2
\end{bmatrix} = A_1B_1 + A_2B_2
$$

Among them:

- $A_1$ and $A_2$ are the left and right parts of the A matrix respectively
- $B_1$ and $B_2$ are the upper and lower parts of the B matrix respectively
- The final result is obtained by $A_1B_1 + A_2B_2$

2. Split B along the columns, A does not need to be split.

$$
A \times B = \begin{bmatrix}
A
\end{bmatrix} \times \begin{bmatrix}
B_1 & B_2
\end{bmatrix} = \begin{bmatrix}
AB_1 & AB_2
\end{bmatrix}
$$

Among them:

- $B_1$ and $B_2$ are the left and right parts of the B matrix respectively
- The final result is obtained by $[AB_1, AB_2]$

At first glance, the difference between the two is not big. Let us further consider that the size of A is $(b, s, h)$ and the size of B is $(h, h')$. Consider the following questions:1. 