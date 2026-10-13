Configuración de .vscode/settings.json (en la raíz)

{
    "python.envFile": "${workspaceFolder}/.env",
    "python.terminal.useEnvFile": true
}

Configuración del .env (raíz)

PYTHONPATH='./src'


# Teacher Forcing:

Consiste en darle al modelo durante el entrenamiento la secuencia correcta completa (add_sub: 5 + 41 = 56) y pedirle que adivine en cada posición que adivine el siguiente token usando solo lo que hay antes.

Todo esto debe funcionar para todos los tokens en paralelo. En la posición del igual no puede ver el 56 aunque esté en la misma secuencia. Solo en evaluación el modelo trabaja sin ayuda, tiene que recibir hasta el igual y producir él solo el 56 y el eos.



https://arxiv.org/pdf/2101.03961
https://arxiv.org/pdf/2401.04088
https://arxiv.org/pdf/1701.06538
