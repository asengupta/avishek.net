<?php get_header(); ?>
			<div id="gray-shading" class="container">
				<div class="padding">
					
					<div id="accordion-wrapper">
						<ul id="accordion"><?php $i = 1; ?>
                          <?php if(have_posts()) : while(have_posts()) : the_post(); ?>
							<li <?php if($i == 1) post_class('first element'); else post_class('element'); $i++; ?>>
								<a href="<?php the_permalink(); ?>" rel="bookmark" title="<?php _e('Permanent Link to') ?> <?php the_title_attribute(); ?>"><?php the_title(); ?><span><?php _e('Written'); ?> <?php the_time('F j, Y') ?> by <?php the_author(); ?> under <?php $category = get_the_category(); if(!empty($category)) echo $category[0]->cat_name; ?></span></a>
								<p>
									<span><?php echo strip_tags(get_the_excerpt()); ?></span>
								</p>
							</li>
							<?php endwhile; ?>
						</ul>
        <div class="navigation">
		  <?php if(function_exists('wp_pagenavi')) : wp_pagenavi();
            elseif (  $wp_query->max_num_pages > 1 ) : ?>
              <div class="left"><?php previous_posts_link('<span>'.__('&laquo; Previous Page').'</span>') ?></div>
              <div class="right"><?php next_posts_link('<span>'.__('Next Page &raquo;').'</span>') ?></div>
            <?php endif; ?>
        </div>
        <?php else: ?>
          <h2 class="center"><?php _e('No Entries Found') ?></h2>
          <p class="center"><?php _e('There are no entries in this blog.') ?></p>
      <?php endif; ?>
					</div>
					
				</div>
                
<?php get_sidebar(); ?>
			</div>
			
            
<?php get_footer(); ?>