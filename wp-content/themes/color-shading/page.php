<?php get_header(); ?>
			<div id="gray-shading" class="container">
				<div class="padding">

					<div id="content">
                            
                      <?php if(have_posts()) : while(have_posts()) : the_post(); ?>

						<div class="gutter">
                        
                          <div <?php post_class(); ?> id="post-<?php the_ID(); ?>">

							<!-- start content -->

							<h2><?php the_title(); ?></h2>
									   
							<?php the_content(); ?>
							<br class="clear" />
                            <?php wp_link_pages('before=<p id="page-links">'.__('Pages:','cover-wp').' &after=</p>'); ?>
							
							<!-- end content -->
							
							<br class="clear" />
                            
                          </div>
							
						</div>
                        
                        <?php if ( comments_open() ) : ?>
                        
						<div class="gutter comments">

                            <?php comments_template(); ?>

                        </div>

                      <?php endif; endwhile; endif; ?>

					</div>
                    
				</div>
			</div>

<?php get_footer(); ?>