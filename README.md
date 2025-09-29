# Landing Page "Power BI para Gestão — Vendas"

Landing page estática, responsiva e offline-ready para divulgar o curso **Power BI para Gestão — Vendas**. Desenvolvida sem dependências externas, com foco em performance, acessibilidade (WCAG AA) e SEO.

## Estrutura de pastas
```
.
├── index.html
├── assets
│   ├── css
│   │   └── styles.css
│   ├── js
│   │   └── main.js
│   └── img
│       ├── img_antes.svg
│       ├── img_depois1.svg
│       └── img_depois2.svg
└── README.md
```

## Como editar o conteúdo
- **Trocar a URL de checkout**: abra `assets/js/main.js` e substitua o valor da constante `KIWIFY_CHECKOUT_URL`. Todos os botões com `data-cta="checkout"` serão atualizados automaticamente.
- **Substituir imagens**: troque os arquivos em `assets/img/` mantendo os mesmos nomes ou atualize os caminhos em `index.html`. Como os placeholders estão em SVG, basta editar o vetor ou substituir por PNG/JPEG otimizados.
- **Atualizar textos**: edite diretamente `index.html`. Os blocos principais estão comentados (`<!-- HERO -->`, `<!-- BENEFÍCIOS -->`, etc.) para facilitar a localização.
- **Ajustar estilos**: `assets/css/styles.css` contém todas as variáveis e componentes. Mantenha o gradiente de fundo para preservar o contraste.

### Checklist antes de publicar
- [ ] Atualizar `KIWIFY_CHECKOUT_URL`
- [ ] Substituir as imagens de antes/depois
- [ ] Revisar e adaptar a copy para sua marca
- [ ] Publicar em ambiente desejado (GitHub Pages, Lovable, etc.)
- [ ] Testar CTAs em desktop e mobile

## Publicação
- **GitHub**: crie um repositório, faça `git add .`, `git commit`, `git remote add origin <URL>` e `git push -u origin main`.
- **Lovable/Kiwify**: compacte os arquivos ou envie-os conforme a plataforma exigir. A página funciona via `file://`, portanto pode ser carregada diretamente.

## Como ativar Order Bump e Upsell na Kiwify (alto nível)
1. Acesse o painel da Kiwify e entre no menu **Produtos**.
2. Selecione o produto do curso e vá em **Order Bump** para adicionar uma oferta complementar (defina título, descrição e preço). Ative ao final.
3. Para **Upsell**, vá em **Funil de vendas** ou **Upsell** e crie um novo passo com a oferta desejada. Configure a página de upsell e a lógica pós-compra.
4. Teste o checkout para garantir que o bump e o upsell aparecem corretamente e levam ao produto configurado.

## Hospedagem offline
Abra `index.html` diretamente no navegador (`file://...`). Todos os assets são locais.

## Boas práticas de manutenção
- Rode auditorias Lighthouse (Performance, Acessibilidade, SEO) sempre que alterar estilos ou scripts.
- Verifique contrastes quando alterar cores ou imagens para manter o nível WCAG AA.
- Utilize `alt` descritivos nas novas imagens e mantenha a hierarquia de headings (H1 → H2 → H3).
